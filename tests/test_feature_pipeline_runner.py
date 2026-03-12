import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.pipeline.runner import DataFeaturePipelineRunner


class _StubExtractor:
    def extract_records(self, subset_root, subset_records, feature_meta, show_progress=True, progress_interval=100):
        del subset_root, feature_meta, show_progress, progress_interval
        return [
            {
                "image_rel": str(subset_records[0]["image_rel"]),
                "annotation_rel": str(subset_records[0]["annotation_rel"]),
                "laplacian_raw": np.asarray([1.0, 2.0], dtype=np.float32),
            }
        ]


class _StubPostprocessor:
    def process_bundle(self, raw_bundle, dimension_schema, progress_interval=100, log_fn=print):
        del progress_interval
        log_fn("[postprocess] stub")
        from clip_dinoiser.feature_utils.data_feature.bundle.processed_bundle import ProcessedFeatureBundle

        return ProcessedFeatureBundle(
            dimension_name=raw_bundle.dimension_name,
            records=[
                {
                    "image_rel": raw_bundle.records[0]["image_rel"],
                    "annotation_rel": raw_bundle.records[0]["annotation_rel"],
                    "schema_version": str(dimension_schema["schema_version"]),
                    "features": {
                        "laplacian": {
                            "encoding": "distribution",
                            "empty_flag": 0,
                            "num_values": 2,
                            "log_num_values": 1.0,
                            "hist": np.asarray([0.5, 0.5], dtype=np.float32),
                            "summary": {},
                            "model_input_fields": ["hist"],
                        }
                    },
                }
            ],
            schema=dimension_schema,
            processing_config={"dimension": raw_bundle.dimension_name},
            summary={"num_samples": 1, "features": {"laplacian": {"empty_samples": 0}}},
        )


class _StubFactory:
    def create_raw_extractor(self, dimension_name, pipeline_config):
        del dimension_name, pipeline_config
        return _StubExtractor()

    def create_postprocessor(self, schema_path):
        del schema_path
        return _StubPostprocessor()

    def load_dimension_schema(self, schema_path, dimension_name):
        del schema_path, dimension_name
        return {
            "schema_version": "quality.v1",
            "features": {
                "laplacian": {
                    "raw_key": "laplacian_raw",
                    "encoding": "distribution",
                    "value_transform": "identity",
                    "num_bins": 2,
                    "range_mode": "fixed",
                    "range_params": {"min": 0.0, "max": 2.0},
                    "summary_fields": {},
                    "model_input_fields": ["hist"],
                }
            },
        }


class TestFeaturePipelineRunner(unittest.TestCase):
    def _make_subset_records(self):
        return [
            {
                "image_rel": "images/train2017/0001.jpg",
                "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
            }
        ]

    def test_pipeline_runner_can_run_raw_stage_with_existing_bundle_outputs(self):
        runner = DataFeaturePipelineRunner(factory=_StubFactory())
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.run_raw(
                dimension_name="quality",
                subset_root="unused",
                subset_records=self._make_subset_records(),
                data_root=tmpdir,
                index_path="data/coco_stuff50k/sample_index.npy",
                feature_meta={},
            )
            self.assertTrue(result["records_path"].endswith("quality_raw_features.npy"))
            self.assertTrue(os.path.exists(result["records_path"]))

    def test_pipeline_runner_can_run_full_stage(self):
        runner = DataFeaturePipelineRunner(factory=_StubFactory())
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.run_full(
                dimension_name="quality",
                subset_root="unused",
                subset_records=self._make_subset_records(),
                data_root=tmpdir,
                index_path="data/coco_stuff50k/sample_index.npy",
                feature_meta={},
                schema_path="docs/feature_schema/unified_processed_feature_schema.json",
            )
            self.assertTrue(result["raw"]["records_path"].endswith("quality_raw_features.npy"))
            self.assertTrue(result["processed"]["records_path"].endswith("quality_processed_features.npy"))
            self.assertTrue(os.path.exists(result["processed"]["records_path"]))


if __name__ == "__main__":
    unittest.main()
