import os
import sys
import unittest


CLIP_DINOISER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CLIP_DINOISER_ROOT, ".."))
if CLIP_DINOISER_ROOT not in sys.path:
    sys.path.insert(0, CLIP_DINOISER_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


from feature.features import get_dataset_feature_spec as get_root_spec
from clip_dinoiser.feature_utils.data_feature.dataset_specs import get_dataset_feature_spec as get_repo_spec


class RootFeatureBridgeTests(unittest.TestCase):
    def test_root_and_repo_dataset_specs_agree_for_voc20(self):
        root_spec = get_root_spec("voc20")
        repo_spec = get_repo_spec("voc20")
        self.assertEqual(root_spec.annotation_suffix, ".png")
        self.assertEqual(repo_spec.annotation_suffix, root_spec.annotation_suffix)
        self.assertEqual(repo_spec.ignore_index, root_spec.ignore_index)
        self.assertEqual(repo_spec.class_names, root_spec.class_names)


if __name__ == "__main__":
    unittest.main()
