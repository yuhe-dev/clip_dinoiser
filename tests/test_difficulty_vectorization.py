import os
import sys
import types
import unittest
from collections import deque
from PIL import Image

import numpy as np
import torch


cv2_stub = sys.modules.get("cv2")
if cv2_stub is None:
    cv2_stub = types.SimpleNamespace()
    sys.modules["cv2"] = cv2_stub

cv2_stub.COLOR_BGR2GRAY = getattr(cv2_stub, "COLOR_BGR2GRAY", 6)
cv2_stub.COLOR_BGR2RGB = getattr(cv2_stub, "COLOR_BGR2RGB", 4)
cv2_stub.CV_64F = getattr(cv2_stub, "CV_64F", np.float64)
cv2_stub.CV_32F = getattr(cv2_stub, "CV_32F", np.float32)
cv2_stub.CC_STAT_AREA = getattr(cv2_stub, "CC_STAT_AREA", 4)


def cvtColor(image, code):
    if image.ndim == 2:
        return image.astype(np.float64)
    if code == cv2_stub.COLOR_BGR2RGB:
        return image[..., ::-1]
    return image[..., :3].mean(axis=2).astype(np.float64)


def connected_components_with_stats(binm, connectivity=8):
    binary = np.asarray(binm, dtype=np.uint8) > 0
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    stats = [[0, 0, 0, 0, 0]]
    centroids = [[0.0, 0.0]]
    label_id = 1
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            queue = deque([(y, x)])
            labels[y, x] = label_id
            coords = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label_id
                        queue.append((ny, nx))
            ys = np.asarray([c[0] for c in coords], dtype=np.float32)
            xs = np.asarray([c[1] for c in coords], dtype=np.float32)
            area = int(len(coords))
            stats.append([int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), area])
            centroids.append([float(xs.mean()), float(ys.mean())])
            label_id += 1

    return label_id, labels, np.asarray(stats, dtype=np.int32), np.asarray(centroids, dtype=np.float32)


cv2_stub.cvtColor = getattr(cv2_stub, "cvtColor", cvtColor)
cv2_stub.connectedComponentsWithStats = getattr(cv2_stub, "connectedComponentsWithStats", connected_components_with_stats)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.implementations.difficulty import (
    EmpiricalDifficultyMaskClip,
    SemanticAmbiguityCLIP,
    SmallObjectRatioCOCOStuff,
)


class ToyClipModel:
    def encode_text(self, tokens):
        if isinstance(tokens, list):
            values = [float(len(text)) for text in tokens]
        else:
            values = [float(tokens)]
        out = torch.tensor([[v, 1.0] for v in values], dtype=torch.float32)
        return out

    def encode_image(self, image_tensor):
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        flat = image_tensor.float().mean(dim=(2, 3))
        if flat.shape[1] == 1:
            flat = torch.cat([flat, torch.ones_like(flat)], dim=1)
        elif flat.shape[1] > 2:
            flat = flat[:, :2]
        return flat


class TestDifficultyVectorization(unittest.TestCase):
    def _make_image(self, h=16, w=16):
        x = np.linspace(0, 255, w, dtype=np.float32)
        y = np.linspace(0, 255, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        base = ((0.7 * xv + 0.3 * yv) % 255).astype(np.uint8)
        return np.stack([base, np.flipud(base), base], axis=-1)

    def _make_small_ratio_mask(self):
        mask = np.full((10, 10), 255, dtype=np.uint8)
        mask[0, 0] = 0
        mask[2:4, 2:4] = 1
        mask[5:10, 5:10] = 2
        return mask

    def _make_region_mask(self):
        mask = np.full((16, 16), 255, dtype=np.uint8)
        mask[1:7, 1:7] = 0
        mask[8:15, 8:15] = 1
        return mask

    def test_small_ratio_vector_score_returns_16_threshold_profile(self):
        metric = SmallObjectRatioCOCOStuff(
            thresholds=np.geomspace(0.001, 0.2, 16).tolist(),
            thing_id_start=0,
            num_things=3,
            default_ignore_index=255,
            use_things_only=False,
        )
        values = metric.get_vector_score(self._make_image(10, 10), mask=self._make_small_ratio_mask())

        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values.shape, (16,))
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))
        self.assertTrue(np.all(values[1:] >= values[:-1]))

    def test_small_ratio_profile_and_count_returns_connected_component_count(self):
        metric = SmallObjectRatioCOCOStuff(
            thresholds=np.geomspace(0.001, 0.2, 16).tolist(),
            thing_id_start=0,
            num_things=3,
            default_ignore_index=255,
            use_things_only=False,
        )

        profile, count = metric.get_profile_and_count(
            self._make_image(10, 10),
            mask=self._make_small_ratio_mask(),
        )

        self.assertEqual(profile.shape, (16,))
        self.assertEqual(count, 3)
        self.assertTrue(np.all(profile[1:] >= profile[:-1]))

    def test_semantic_gap_vector_score_returns_region_gap_values(self):
        metric = SemanticAmbiguityCLIP(
            clip_model=ToyClipModel(),
            tokenizer=lambda texts: texts,
            preprocess=lambda crop: torch.from_numpy(np.asarray(crop).copy().transpose(2, 0, 1)).float(),
            device="cpu",
            thing_id_start=0,
            num_things=2,
            default_ignore_index=255,
            use_things_only=False,
            min_region_pixels=4,
            max_regions_per_image=10,
        )
        values = metric.get_vector_score(
            self._make_image(),
            mask=self._make_region_mask(),
            meta={"class_names": ["cat", "dog"]},
        )

        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values.ndim, 1)
        self.assertEqual(values.shape[0], 2)
        self.assertTrue(np.all(values >= 0.0))

    def test_semantic_gap_preprocess_accepts_pil_images(self):
        def pil_preprocess(crop):
            self.assertIsInstance(crop, Image.Image)
            arr = np.asarray(crop, dtype=np.float32)
            return torch.from_numpy(arr.transpose(2, 0, 1))

        metric = SemanticAmbiguityCLIP(
            clip_model=ToyClipModel(),
            tokenizer=lambda texts: texts,
            preprocess=pil_preprocess,
            device="cpu",
            thing_id_start=0,
            num_things=2,
            default_ignore_index=255,
            use_things_only=False,
            min_region_pixels=4,
            max_regions_per_image=10,
        )
        values = metric.get_vector_score(
            self._make_image(),
            mask=self._make_region_mask(),
            meta={"class_names": ["cat", "dog"]},
        )

        self.assertEqual(values.shape[0], 2)

    def test_empirical_iou_vector_score_returns_per_class_ious(self):
        gt_mask = np.full((4, 4), 255, dtype=np.uint8)
        gt_mask[:2, :2] = 0
        gt_mask[2:, 2:] = 1

        pred_mask = np.full((4, 4), 255, dtype=np.int32)
        pred_mask[:2, :2] = 0
        pred_mask[2:, 1:] = 1

        metric = EmpiricalDifficultyMaskClip(
            predictor=lambda image, meta=None: pred_mask,
            default_ignore_index=255,
        )
        values = metric.get_vector_score(self._make_image(4, 4), mask=gt_mask)

        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))


if __name__ == "__main__":
    unittest.main()
