import os
import sys
import unittest

import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.helpers.trainability import build_optimizer_groups, configure_trainable_modules, resolve_module


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_proj = nn.Linear(4, 4)
        self.bkg_decoder = nn.Linear(4, 4)
        self.clip_backbone = nn.Module()
        self.clip_backbone.decode_head = _ToyBlock()
        self.clip_backbone.backbone = nn.Module()
        self.clip_backbone.backbone.visual = nn.Module()
        self.clip_backbone.backbone.visual.transformer = nn.Module()
        self.clip_backbone.backbone.visual.transformer.resblocks = nn.ModuleList([_ToyBlock(), _ToyBlock()])
        self.clip_backbone.backbone.visual.ln_post = nn.LayerNorm(4)


class ResearchTrainabilityTests(unittest.TestCase):
    def test_trainability_resolves_negative_block_index_and_optimizer_groups(self):
        model = _ToyModel()
        configured = configure_trainable_modules(
            model,
            [
                "obj_proj",
                "bkg_decoder",
                "clip_backbone.decode_head.proj",
                "clip_backbone.backbone.visual.transformer.resblocks.-1",
                "clip_backbone.backbone.visual.ln_post",
            ],
        )
        self.assertIn("clip_backbone.backbone.visual.transformer.resblocks.-1", configured)
        resolved = resolve_module(model, "clip_backbone.backbone.visual.transformer.resblocks.-1")
        self.assertIsInstance(resolved, _ToyBlock)

        optimizer_groups, summary = build_optimizer_groups(
            model,
            corr_lr=5e-5,
            found_lr=1e-2,
            backbone_lr=1e-5,
        )
        self.assertEqual(len(optimizer_groups), 3)
        self.assertTrue(any(name.startswith("bkg_decoder.") for name in summary.found_param_names))
        self.assertTrue(any(name.startswith("clip_backbone.backbone.") for name in summary.backbone_param_names))


if __name__ == "__main__":
    unittest.main()
