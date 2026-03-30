# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology & Oriane Simeoni, valeo.ai
# ---------------------------------------------------------------------------------------------------

import os
from typing import Optional

import cv2
import numpy as np
from mmengine.registry import init_default_scope
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import ImageToTensor, to_tensor

init_default_scope('mmseg')


@PIPELINES.register_module()
class LoadNpyImageFromFile:
    def __init__(self, to_float32: bool = False) -> None:
        self.to_float32 = bool(to_float32)

    def __call__(self, results):
        return self.transform(results)

    def transform(self, results: dict) -> Optional[dict]:
        img_info = dict(results.get("img_info") or {})
        img_prefix = results.get("img_prefix")
        cache_filename = img_info.get("cache_filename") or img_info.get("filename")
        if cache_filename is None:
            raise KeyError("img_info must contain 'cache_filename' or 'filename'")
        filename = str(cache_filename)
        if img_prefix is not None and not os.path.isabs(filename):
            filename = os.path.join(str(img_prefix), filename)

        img = np.load(filename)
        if img.ndim < 3:
            img = np.expand_dims(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        img_shape = tuple(int(v) for v in img.shape)
        ori_shape = tuple(int(v) for v in img_info.get("ori_shape", img_shape))
        scale_factor = tuple(float(v) for v in img_info.get("scale_factor", (1.0, 1.0, 1.0, 1.0)))

        results["filename"] = filename
        results["ori_filename"] = str(img_info.get("filename", os.path.basename(filename)))
        results["img"] = img
        results["img_shape"] = img_shape
        results["ori_shape"] = ori_shape
        results["pad_shape"] = img_shape
        results["scale_factor"] = scale_factor
        results["img_fields"] = ["img"]
        channel_count = int(img.shape[2])
        results["img_norm_cfg"] = dict(
            mean=np.zeros(channel_count, dtype=np.float32),
            std=np.ones(channel_count, dtype=np.float32),
            to_rgb=False,
        )
        return results


@PIPELINES.register_module()
class ToRGB:
    def __call__(self, results):
        return self.transform(results)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to go from BGR to RGB.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        results['img'] = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
        return results


# ________________
# Modified version from mmcv to directly convert a tensor to float
# MAKE SURE YOU USE IT ONLY FOR IMAGES

@PIPELINES.register_module()
class ImageToTensorV2(ImageToTensor):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Required keys:

    - all these keys in `keys`

    Modified Keys:

    - all these keys in `keys`

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys: dict) -> None:
        super(ImageToTensorV2, self).__init__(keys)
        self.keys = keys

    def __call__(self, results):
        return self.transform(results)

    def transform(self, results: dict) -> dict:
        """Transform function to convert image in results to
        :obj:`torch.Tensor` and transpose the channel order.
        Args:
            results (dict): Result dict contains the image data to convert.
        Returns:
            dict: The result dict contains the image converted
            to :obj:``torch.Tensor`` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            results[key] = (to_tensor(img.copy()).permute(2, 0, 1)).contiguous() / 255.

        return results
