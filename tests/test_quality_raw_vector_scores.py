import os
import sys
import types
import unittest

import numpy as np


cv2_stub = sys.modules.get("cv2")
if cv2_stub is None:
    cv2_stub = types.SimpleNamespace()
    sys.modules["cv2"] = cv2_stub

cv2_stub.COLOR_BGR2GRAY = getattr(cv2_stub, "COLOR_BGR2GRAY", 6)
cv2_stub.CV_64F = getattr(cv2_stub, "CV_64F", np.float64)
cv2_stub.CV_32F = getattr(cv2_stub, "CV_32F", np.float32)
cv2_stub.NORM_MINMAX = getattr(cv2_stub, "NORM_MINMAX", 32)


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
    return out.astype(dtype)


def Sobel(gray, dtype, dx, dy, ksize=3):
    g = gray.astype(np.float64)
    out = np.zeros_like(g)
    if dx == 1 and dy == 0:
        out[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    elif dx == 0 and dy == 1:
        out[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return out.astype(dtype)


def magnitude(gx, gy):
    return np.sqrt(np.asarray(gx, dtype=np.float64) ** 2 + np.asarray(gy, dtype=np.float64) ** 2)


def normalize(src, dst, alpha, beta, norm_type):
    src_arr = np.asarray(src, dtype=np.float64)
    s_min = float(src_arr.min()) if src_arr.size else 0.0
    s_max = float(src_arr.max()) if src_arr.size else 0.0
    if s_max <= s_min:
        dst[...] = 0.0
        return dst
    dst[...] = (src_arr - s_min) / (s_max - s_min)
    dst[...] = dst * (beta - alpha) + alpha
    return dst


def _pad_edge(array):
    return np.pad(array, ((1, 1), (1, 1)), mode="edge")


def dilate(image, kernel, iterations=1):
    out = image.astype(np.uint8)
    for _ in range(iterations):
        padded = _pad_edge(out)
        windows = [
            padded[i : i + out.shape[0], j : j + out.shape[1]]
            for i in range(3)
            for j in range(3)
        ]
        out = np.maximum.reduce(windows)
    return out


def erode(image, kernel, iterations=1):
    out = image.astype(np.uint8)
    for _ in range(iterations):
        padded = _pad_edge(out)
        windows = [
            padded[i : i + out.shape[0], j : j + out.shape[1]]
            for i in range(3)
            for j in range(3)
        ]
        out = np.minimum.reduce(windows)
    return out


def absdiff(a, b):
    return np.abs(np.asarray(a, dtype=np.int32) - np.asarray(b, dtype=np.int32)).astype(np.uint8)


cv2_stub.cvtColor = getattr(cv2_stub, "cvtColor", cvtColor)
cv2_stub.Laplacian = getattr(cv2_stub, "Laplacian", Laplacian)
cv2_stub.Sobel = getattr(cv2_stub, "Sobel", Sobel)
cv2_stub.magnitude = getattr(cv2_stub, "magnitude", magnitude)
cv2_stub.normalize = getattr(cv2_stub, "normalize", normalize)
cv2_stub.dilate = getattr(cv2_stub, "dilate", dilate)
cv2_stub.erode = getattr(cv2_stub, "erode", erode)
cv2_stub.absdiff = getattr(cv2_stub, "absdiff", absdiff)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.implementations.quality import (
    BoundaryGradientAdherence,
    WeakTexturePCANoise,
)


class TestQualityRawVectorScores(unittest.TestCase):
    def _make_image(self, h=32, w=32):
        x = np.linspace(0, 255, w, dtype=np.float32)
        y = np.linspace(0, 255, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        base = ((xv + yv) * 0.5).astype(np.uint8)
        return np.stack([base, base, base], axis=-1)

    def _make_mask(self, h=32, w=32):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        return mask

    def test_noise_pca_vector_score_returns_patchwise_raw_values(self):
        metric = WeakTexturePCANoise(
            patch_size=8,
            stride=8,
            weak_texture_percentile=50.0,
            min_patches_for_pca=1,
            tail_eig_k=1,
        )
        values = metric.get_vector_score(self._make_image())

        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values.ndim, 1)
        self.assertEqual(values.shape, (16,))
        self.assertTrue(np.all(values >= 0.0))

    def test_bga_vector_score_returns_boundary_gradient_values(self):
        metric = BoundaryGradientAdherence()
        values = metric.get_vector_score(self._make_image(), mask=self._make_mask())

        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values.ndim, 1)
        self.assertGreater(values.shape[0], 0)
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_bga_vector_score_returns_empty_when_mask_missing(self):
        metric = BoundaryGradientAdherence()
        values = metric.get_vector_score(self._make_image(), mask=None)

        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values.shape, (0,))


if __name__ == "__main__":
    unittest.main()
