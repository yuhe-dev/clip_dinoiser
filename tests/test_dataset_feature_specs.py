import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.dataset_specs import (
    get_dataset_feature_spec,
    list_dataset_feature_specs,
    merge_feature_meta_with_dataset_spec,
)


class DatasetFeatureSpecTests(unittest.TestCase):
    def test_registry_lists_expected_specs(self):
        names = list_dataset_feature_specs()
        self.assertIn("coco_stuff", names)
        self.assertIn("voc20", names)
        self.assertIn("cityscapes", names)

    def test_voc20_spec_uses_raw_voc_mask_protocol(self):
        spec = get_dataset_feature_spec("voc20")
        self.assertEqual(spec.ignore_index, 255)
        self.assertTrue(spec.reduce_zero_label)
        self.assertEqual(spec.annotation_suffix, ".png")
        self.assertEqual(spec.thing_ids[0], 1)
        self.assertEqual(spec.thing_ids[-1], 20)

    def test_merge_feature_meta_with_dataset_spec_preserves_overrides(self):
        merged = merge_feature_meta_with_dataset_spec({"patch_size": 32}, dataset_name="voc20")
        self.assertEqual(merged["patch_size"], 32)
        self.assertEqual(merged["ignore_index"], 255)
        self.assertEqual(merged["annotation_suffix"], ".png")
        self.assertEqual(len(merged["class_names"]), 20)


if __name__ == "__main__":
    unittest.main()
