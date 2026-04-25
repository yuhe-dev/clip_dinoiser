import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.voc_feature_prep import (
    DEFAULT_AXIS_NAMES,
    build_voc_train_aug_records,
    compute_voc_feature_rows,
    prepare_voc_train_aug_feature_experiment,
)


class VocFeatureSubsetPreparationTests(unittest.TestCase):
    def _build_toy_voc_train_aug_root(self, root: Path) -> Path:
        source_root = root / "VOC2012"
        (source_root / "JPEGImages").mkdir(parents=True)
        (source_root / "SegmentationClassAug").mkdir(parents=True)
        (source_root / "ImageSets" / "Segmentation").mkdir(parents=True)

        stems = [f"2007_{index:06d}" for index in range(1, 9)]
        masks = [
            np.asarray([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
            np.asarray([[0, 1, 0, 2], [0, 1, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
            np.asarray([[0, 20, 20, 0], [0, 20, 20, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
            np.asarray([[0, 19, 0, 0], [0, 0, 0, 0], [3, 3, 3, 0], [0, 0, 0, 0]], dtype=np.uint8),
            np.asarray([[4, 0, 4, 0], [0, 0, 0, 0], [5, 0, 5, 0], [0, 0, 0, 0]], dtype=np.uint8),
            np.asarray([[6, 6, 6, 6], [0, 0, 0, 0], [0, 7, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
            np.asarray([[8, 0, 0, 9], [0, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
            np.asarray([[11, 12, 0, 0], [0, 13, 0, 0], [0, 0, 14, 0], [0, 0, 0, 15]], dtype=np.uint8),
        ]

        for stem, mask in zip(stems, masks):
            (source_root / "JPEGImages" / f"{stem}.jpg").write_bytes(f"img-{stem}".encode("utf-8"))
            Image.fromarray(mask).save(source_root / "SegmentationClassAug" / f"{stem}.png")

        (source_root / "ImageSets" / "Segmentation" / "train_aug.txt").write_text(
            "\n".join(stems) + "\n",
            encoding="utf-8",
        )
        return source_root

    def test_build_voc_train_aug_records_reads_aug_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = self._build_toy_voc_train_aug_root(Path(tmpdir))
            records = build_voc_train_aug_records(source_root)

            self.assertEqual(len(records), 8)
            self.assertEqual(records[0].image_rel, "JPEGImages/2007_000001.jpg")
            self.assertEqual(records[0].annotation_rel, "SegmentationClassAug/2007_000001.png")

    def test_default_axes_remain_screening_defaults(self):
        self.assertEqual(DEFAULT_AXIS_NAMES, ("small_object_ratio", "rare_class_coverage"))

    def test_prepare_voc_train_aug_feature_experiment_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = self._build_toy_voc_train_aug_root(root)
            output_dir = root / "artifacts"

            payload = prepare_voc_train_aug_feature_experiment(
                data_root=str(source_root),
                output_dir=str(output_dir),
                subset_size=2,
                anchor_seed=0,
                candidate_budget=4,
                small_object_tau_ratio=0.2,
            )

            feature_table_path = Path(payload["feature_table_path"])
            summary_path = Path(payload["summary_path"])
            feasibility_report_path = Path(payload["feasibility_report_path"])
            manifest_index_path = Path(payload["manifest_index_path"])
            self.assertTrue(feature_table_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(feasibility_report_path.is_file())
            self.assertTrue(manifest_index_path.is_file())

            lines = feature_table_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 8)
            first_row = json.loads(lines[0])
            self.assertIn("small_object_ratio", first_row)
            self.assertIn("rare_class_coverage", first_row)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["pool_size"], 8)
            self.assertIn("small_object_ratio", summary["axis_summary"])
            self.assertIn("rare_class_coverage", summary["axis_summary"])
            self.assertIn("mid_mean", summary["axis_summary"]["small_object_ratio"])
            self.assertIn("bucket_capacity", summary["axis_summary"]["small_object_ratio"])
            self.assertEqual(
                summary["axis_summary"]["small_object_ratio"]["bucket_capacity"]["method"],
                "rank_tertile",
            )
            self.assertIn("feasibility_gate", summary)
            feasibility = json.loads(feasibility_report_path.read_text(encoding="utf-8"))
            self.assertIn("axis_correlations", feasibility)
            self.assertIn("axis_rank_correlations", feasibility)
            self.assertIn("small_object_ratio", feasibility["axes"])

            manifest_index = json.loads(manifest_index_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest_index), 9)
            anchor_manifest = json.loads(Path(manifest_index["anchor"]).read_text(encoding="utf-8"))
            self.assertEqual(anchor_manifest["candidate_id"], "voc_train_aug_anchor_2_seed0")
            self.assertEqual(len(anchor_manifest["sample_ids"]), 2)
            self.assertEqual(len(anchor_manifest["sample_paths"]), 2)

            high_manifest = json.loads(
                Path(manifest_index["small_object_ratio.high"]).read_text(encoding="utf-8")
            )
            low_manifest = json.loads(
                Path(manifest_index["small_object_ratio.low"]).read_text(encoding="utf-8")
            )
            mid_manifest = json.loads(
                Path(manifest_index["small_object_ratio.mid"]).read_text(encoding="utf-8")
            )
            matched_manifest = json.loads(
                Path(manifest_index["small_object_ratio.matched_random"]).read_text(encoding="utf-8")
            )
            anchor_ids = set(anchor_manifest["sample_ids"])
            high_ids = set(high_manifest["sample_ids"])
            low_ids = set(low_manifest["sample_ids"])
            mid_ids = set(mid_manifest["sample_ids"])
            matched_ids = set(matched_manifest["sample_ids"])
            self.assertFalse(anchor_ids & matched_ids)
            self.assertFalse(high_ids & matched_ids)
            self.assertFalse(low_ids & matched_ids)
            self.assertEqual(len(mid_ids), 2)
            overlap_counts = summary["axis_summary"]["small_object_ratio"]["overlap_counts"]
            self.assertEqual(overlap_counts["anchor"], 0)
            self.assertEqual(overlap_counts["high"], 0)
            self.assertEqual(overlap_counts["low"], 0)
            self.assertIn("mid", overlap_counts)
            gate = summary["feasibility_gate"]["small_object_ratio"]
            self.assertEqual(gate["matched_random_mid_overlap"], overlap_counts["mid"])

    def test_prepare_voc_train_aug_feature_experiment_can_limit_axes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = self._build_toy_voc_train_aug_root(root)
            output_dir = root / "artifacts"

            payload = prepare_voc_train_aug_feature_experiment(
                data_root=str(source_root),
                output_dir=str(output_dir),
                subset_size=2,
                anchor_seed=0,
                candidate_budget=4,
                feature_axes=["rare_class_coverage"],
            )

            summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["feature_axes"], ["rare_class_coverage"])
            self.assertEqual(summary["feature_axes"], ["rare_class_coverage"])
            self.assertEqual(set(summary["axis_summary"].keys()), {"rare_class_coverage"})

            rows = [
                json.loads(line)
                for line in Path(payload["feature_table_path"]).read_text(encoding="utf-8").strip().splitlines()
            ]
            self.assertIn("rare_class_coverage", rows[0])
            self.assertNotIn("small_object_ratio", rows[0])

    def test_prepare_voc_train_aug_feature_experiment_supports_new_mask_native_axes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = self._build_toy_voc_train_aug_root(root)
            output_dir = root / "artifacts"

            payload = prepare_voc_train_aug_feature_experiment(
                data_root=str(source_root),
                output_dir=str(output_dir),
                subset_size=2,
                anchor_seed=0,
                candidate_budget=4,
                feature_axes=[
                    "foreground_class_count",
                    "pixel_class_entropy",
                    "foreground_area_ratio",
                    "foreground_component_count",
                    "component_fragmentation",
                ],
            )

            summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                set(summary["axis_summary"].keys()),
                {
                    "foreground_class_count",
                    "pixel_class_entropy",
                    "foreground_area_ratio",
                    "foreground_component_count",
                    "component_fragmentation",
                },
            )

            rows = [
                json.loads(line)
                for line in Path(payload["feature_table_path"]).read_text(encoding="utf-8").strip().splitlines()
            ]
            first_row = rows[0]
            self.assertIn("foreground_class_count", first_row)
            self.assertIn("pixel_class_entropy", first_row)
            self.assertIn("foreground_area_ratio", first_row)
            self.assertIn("foreground_component_count", first_row)
            self.assertIn("component_fragmentation", first_row)

    def test_compute_voc_feature_rows_supports_probe_stage_axes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = self._build_toy_voc_train_aug_root(root)
            records = build_voc_train_aug_records(source_root)

            result = compute_voc_feature_rows(
                records,
                data_root=str(source_root),
                feature_axes=[
                    "rare_class_coverage",
                    "rare_class_exposure_clipped",
                    "crop_survival_score",
                ],
                rare_class_clip_percentile=75.0,
                crop_survival_simulations=3,
                crop_survival_seed=13,
            )

            self.assertIn("rare_class_coverage", result.axis_scores)
            self.assertIn("rare_class_exposure_clipped", result.axis_scores)
            self.assertIn("crop_survival_score", result.axis_scores)
            np.testing.assert_allclose(
                result.axis_scores["rare_class_coverage"],
                (result.class_presence_matrix.astype(np.float32) * result.rarity_weights[None, :]).sum(axis=1),
            )
            clipped_positive = result.clipped_rarity_weights[result.clipped_rarity_weights > 0]
            self.assertAlmostEqual(float(clipped_positive.mean()), 1.0, places=6)
            self.assertTrue(np.all(np.isfinite(result.clipped_rarity_weights)))
            self.assertTrue(np.all(result.axis_scores["crop_survival_score"] >= 0.0))
            self.assertTrue(np.all(result.axis_scores["crop_survival_score"] <= 1.0))
            self.assertIn("rare_class_exposure_clipped", result.rows[0])
            self.assertIn("crop_survival_score", result.rows[0])


if __name__ == "__main__":
    unittest.main()
