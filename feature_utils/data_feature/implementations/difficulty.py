import numpy as np
from typing import Optional, Dict, Any, Iterable, List
from ..dimensions import DifficultyDimension
import torch
import torch.nn.functional as F
from PIL import Image
from collections import deque

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    class _CV2Fallback:
        COLOR_BGR2RGB = 4
        CC_STAT_AREA = 4

        @staticmethod
        def cvtColor(image: np.ndarray, code: int) -> np.ndarray:
            if code != _CV2Fallback.COLOR_BGR2RGB:
                raise ValueError(f"Unsupported fallback cvtColor code: {code}")
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError("Fallback cvtColor expects an HxWx3 image array.")
            return image[..., ::-1]

        @staticmethod
        def connectedComponentsWithStats(binm: np.ndarray, connectivity: int = 8):
            binary = np.asarray(binm, dtype=np.uint8) > 0
            h, w = binary.shape
            labels = np.zeros((h, w), dtype=np.int32)
            stats = [[0, 0, 0, 0, 0]]
            centroids = [[0.0, 0.0]]
            label_id = 1
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if int(connectivity) == 8:
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
                    ys = np.asarray([coord[0] for coord in coords], dtype=np.float32)
                    xs = np.asarray([coord[1] for coord in coords], dtype=np.float32)
                    stats.append(
                        [
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max() - xs.min() + 1),
                            int(ys.max() - ys.min() + 1),
                            int(len(coords)),
                        ]
                    )
                    centroids.append([float(xs.mean()), float(ys.mean())])
                    label_id += 1

            return (
                label_id,
                labels,
                np.asarray(stats, dtype=np.int32),
                np.asarray(centroids, dtype=np.float32),
            )

    cv2 = _CV2Fallback()


class SmallObjectRatioCOCOStuff(DifficultyDimension):
    """
    COCO-Stuff small object ratio (thing-only by default) using connected components.

    - Works on semantic label maps (trainIds).
    - Default assumes COCO-Stuff 164k labelTrainIds:
        thing_ids = [0..79], stuff_ids = [80..170], ignore_index = 255
      (You can override via meta or constructor.)

    Score:
      area-weighted fraction of pixels that belong to "small" connected components.
    """
    def __init__(
        self,
        tau_ratio: float = 0.02,
        connectivity: int = 8,
        thing_id_start: int = 0,   # 164k: 0, 10k: 1
        num_things: int = 80,
        default_ignore_index: Optional[int] = 255,  # 164k: 255, 10k often: 0
        use_things_only: bool = True,
        thresholds: Optional[List[float]] = None,
    ):
        super().__init__("small_object_ratio")
        self.tau_ratio = float(tau_ratio)
        self.connectivity = int(connectivity)
        self.thing_id_start = int(thing_id_start)
        self.num_things = int(num_things)
        self.default_ignore_index = default_ignore_index
        self.use_things_only = bool(use_things_only)
        if thresholds is None:
            thresholds = np.geomspace(0.001, 0.05, 16).astype(np.float32).tolist()
        self.thresholds = [float(v) for v in thresholds]

    def _infer_ignore_index(self, m: np.ndarray, meta: Dict[str, Any]) -> Optional[int]:
        # Priority: meta > constructor default > auto-detect
        if "ignore_index" in meta and meta["ignore_index"] is not None:
            return int(meta["ignore_index"])
        if self.default_ignore_index is not None:
            return int(self.default_ignore_index)
        # auto-detect common values
        vals = np.unique(m)
        if 255 in vals:
            return 255
        if 0 in vals and vals.max() > 170:
            return 0
        return None

    def _default_thing_ids(self) -> List[int]:
        return list(range(self.thing_id_start, self.thing_id_start + self.num_things))

    def _collect_area_ratios(
        self,
        mask: Optional[np.ndarray],
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        if mask is None:
            return np.asarray([], dtype=np.float32)
        meta = meta or {}

        m = mask.astype(np.int32)
        H, W = m.shape[:2]
        image_area = float(max(H * W, 1))
        ignore_index = self._infer_ignore_index(m, meta)
        valid = (m != ignore_index) if ignore_index is not None else np.ones_like(m, dtype=bool)

        if meta.get("class_ids", None) is not None:
            use_ids: Iterable[int] = meta["class_ids"]
        elif self.use_things_only:
            use_ids = meta.get("thing_ids", None) or self._default_thing_ids()
        else:
            use_ids = np.unique(m[valid]).tolist()

        area_ratios: List[float] = []
        for cid in use_ids:
            cid = int(cid)
            if ignore_index is not None and cid == int(ignore_index):
                continue
            binm = (m == cid) & valid
            if not binm.any():
                continue
            num, _, stats, _ = cv2.connectedComponentsWithStats(
                binm.astype(np.uint8),
                connectivity=self.connectivity
            )
            if num <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
            area_ratios.extend((areas / image_area).tolist())

        return np.asarray(area_ratios, dtype=np.float32)

    def get_profile_and_count(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> tuple[np.ndarray, int]:
        ratios = self._collect_area_ratios(mask=mask, meta=meta)
        if ratios.size == 0:
            return np.zeros((len(self.thresholds),), dtype=np.float32), 0
        values = [
            float(np.mean(ratios < threshold))
            for threshold in self.thresholds
        ]
        return np.asarray(values, dtype=np.float32), int(ratios.size)

    def get_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> float:
        if mask is None:
            return 0.0
        meta = meta or {}

        m = mask.astype(np.int32)
        H, W = m.shape[:2]
        tau = max(1, int(self.tau_ratio * H * W))
        ignore_index = self._infer_ignore_index(m, meta)
        valid = (m != ignore_index) if ignore_index is not None else np.ones_like(m, dtype=bool)

        # Decide which ids to consider
        if meta.get("class_ids", None) is not None:
            use_ids: Iterable[int] = meta["class_ids"]
        elif self.use_things_only:
            use_ids = meta.get("thing_ids", None) or self._default_thing_ids()
        else:
            # all present labels (excluding ignore)
            use_ids = np.unique(m[valid]).tolist()

        total_area = 0
        small_area = 0

        for cid in use_ids:
            cid = int(cid)
            if ignore_index is not None and cid == int(ignore_index):
                continue
            binm = (m == cid) & valid
            if not binm.any():
                continue
            num, _, stats, _ = cv2.connectedComponentsWithStats(
                binm.astype(np.uint8),
                connectivity=self.connectivity
            )
            if num <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
            total_area += len(areas)
            small_area += int((areas < tau).sum())

        if total_area == 0:
            return 0.0
        return float(small_area / (total_area + 1e-8))

    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        profile, _ = self.get_profile_and_count(image=image, mask=mask, meta=meta)
        return profile
    
class SemanticAmbiguityCLIP(DifficultyDimension):
    """
    Semantic Ambiguity (Visual-Semantic Gap) for OVS:
    Use CLIP to measure how well mask-region visual features align with its label text.

    Output:
      - A scalar ambiguity score in [0, 2] roughly (since 1-cos in [0,2]).
        Larger => more ambiguous / less aligned.

    Required meta:
      - "class_names": List[str] indexed by trainId (same order as dataset CLASSES)
      - optionally "thing_ids", "ignore_index"
      - optionally "id_to_text": Dict[int, str]  (override class name -> text prompt)
    """

    def __init__(
        self,
        clip_model,               # your CLIP wrapper/model
        tokenizer=None,           # tokenizer if needed
        device: str = "cuda",
        prompt_template: str = "a photo of a {}",
        thing_id_start: int = 0,
        num_things: int = 80,
        default_ignore_index: Optional[int] = 255,
        use_things_only: bool = True,
        min_region_pixels: int = 256,     # ignore tiny regions to reduce noise
        crop_expand: float = 0.1,         # expand bbox for context
        max_regions_per_image: int = 20,  # cap for speed
        preprocess=None,
    ):
        super().__init__("semantic_ambiguity_clip")
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_template = prompt_template
        self.thing_id_start = int(thing_id_start)
        self.num_things = int(num_things)
        self.default_ignore_index = default_ignore_index
        self.use_things_only = bool(use_things_only)
        self.min_region_pixels = int(min_region_pixels)
        self.crop_expand = float(crop_expand)
        self.max_regions_per_image = int(max_regions_per_image)
        self.preprocess = preprocess

        self._text_cache: Dict[str, torch.Tensor] = {}

    def _infer_ignore_index(self, m: np.ndarray, meta: Dict[str, Any]) -> Optional[int]:
        if "ignore_index" in meta and meta["ignore_index"] is not None:
            return int(meta["ignore_index"])
        if self.default_ignore_index is not None:
            return int(self.default_ignore_index)
        vals = np.unique(m)
        if 255 in vals:
            return 255
        if 0 in vals and vals.max() > 170:
            return 0
        return None

    def _default_thing_ids(self) -> List[int]:
        return list(range(self.thing_id_start, self.thing_id_start + self.num_things))

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        if text in self._text_cache:
            return self._text_cache[text]

        # ---- You must adapt this block to your CLIP implementation ----
        # Expected: return a normalized embedding vector [D]
        if hasattr(self.clip_model, "encode_text"):
            # open_clip style
            if self.tokenizer is not None:
                tokens = self.tokenizer([text])
                if hasattr(tokens, "to"):
                    tokens = tokens.to(self.device)
            else:
                tokens = text
            t = self.clip_model.encode_text(tokens)
        else:
            # transformers style placeholder
            raise RuntimeError("Please adapt _encode_text() to your CLIP implementation.")
        t = t.float()
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
        t = t.squeeze(0)
        self._text_cache[text] = t
        return t

    @torch.no_grad()
    def _encode_image_crop(self, crop_rgb: np.ndarray) -> torch.Tensor:
        if not hasattr(self.clip_model, "encode_image"):
            raise RuntimeError("clip_model must provide encode_image().")
        crop_rgb = np.ascontiguousarray(crop_rgb)

        if self.preprocess is not None:
            image_tensor = self.preprocess(Image.fromarray(crop_rgb))
        else:
            image_tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float() / 255.0

        if not torch.is_tensor(image_tensor):
            image_tensor = torch.as_tensor(image_tensor)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        v = self.clip_model.encode_image(image_tensor)
        v = v.float()
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
        return v.squeeze(0)

    def _bbox_from_mask(self, binm: np.ndarray):
        ys, xs = np.where(binm)
        if ys.size == 0:
            return None
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return int(x0), int(y0), int(x1), int(y1)

    def _expand_bbox(self, bbox, W: int, H: int):
        x0, y0, x1, y1 = bbox
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        ex = int(self.crop_expand * bw)
        ey = int(self.crop_expand * bh)
        x0 = max(0, x0 - ex)
        y0 = max(0, y0 - ey)
        x1 = min(W - 1, x1 + ex)
        y1 = min(H - 1, y1 + ey)
        return x0, y0, x1, y1

    def get_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Returns a scalar ambiguity score. Larger = more ambiguous.
        """
        if mask is None or image is None:
            return 0.0
        meta = meta or {}

        m = mask.astype(np.int32)
        H, W = m.shape[:2]
        ignore_index = self._infer_ignore_index(m, meta)
        valid = (m != ignore_index) if ignore_index is not None else np.ones_like(m, dtype=bool)

        # class name list is required
        class_names = meta.get("class_names", None)
        if class_names is None:
            raise ValueError("SemanticAmbiguityCLIP requires meta['class_names'] (aligned with trainIds).")

        # which ids to consider
        if meta.get("class_ids", None) is not None:
            use_ids: Iterable[int] = meta["class_ids"]
        elif self.use_things_only:
            use_ids = meta.get("thing_ids", None) or self._default_thing_ids()
        else:
            use_ids = np.unique(m[valid]).tolist()

        # Convert image to RGB once
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Collect candidate regions
        regions = []
        for cid in use_ids:
            cid = int(cid)
            if ignore_index is not None and cid == int(ignore_index):
                continue
            binm = (m == cid) & valid
            area = int(binm.sum())
            if area < self.min_region_pixels:
                continue
            bbox = self._bbox_from_mask(binm)
            if bbox is None:
                continue
            bbox = self._expand_bbox(bbox, W=W, H=H)
            regions.append((cid, area, bbox))

        if len(regions) == 0:
            return 0.0

        # For speed: take largest regions first
        regions.sort(key=lambda x: x[1], reverse=True)
        regions = regions[: self.max_regions_per_image]

        # Build ambiguity score
        total_w = 0.0
        total_gap = 0.0

        # optional override for prompt text
        id_to_text = meta.get("id_to_text", {}) or {}

        for cid, area, (x0, y0, x1, y1) in regions:
            name = class_names[cid]
            text = id_to_text.get(cid, self.prompt_template.format(name))

            # text embedding
            t = self._encode_text(text)

            # crop embedding (you must implement _encode_image_crop for your CLIP)
            crop = rgb[y0:y1+1, x0:x1+1, :]
            v = self._encode_image_crop(crop)

            # gap = 1 - cosine similarity
            gap = float(1.0 - torch.sum(v * t).item())
            w = float(area)

            total_gap += w * gap
            total_w += w

        return float(total_gap / (total_w + 1e-8))

    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        if mask is None or image is None:
            return np.asarray([], dtype=np.float32)
        meta = meta or {}

        m = mask.astype(np.int32)
        H, W = m.shape[:2]
        ignore_index = self._infer_ignore_index(m, meta)
        valid = (m != ignore_index) if ignore_index is not None else np.ones_like(m, dtype=bool)
        class_names = meta.get("class_names", None)
        if class_names is None:
            raise ValueError("SemanticAmbiguityCLIP requires meta['class_names'] (aligned with trainIds).")

        if meta.get("class_ids", None) is not None:
            use_ids: Iterable[int] = meta["class_ids"]
        elif self.use_things_only:
            use_ids = meta.get("thing_ids", None) or self._default_thing_ids()
        else:
            use_ids = np.unique(m[valid]).tolist()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        id_to_text = meta.get("id_to_text", {}) or {}
        regions = []
        for cid in use_ids:
            cid = int(cid)
            if ignore_index is not None and cid == int(ignore_index):
                continue
            binm = (m == cid) & valid
            area = int(binm.sum())
            if area < self.min_region_pixels:
                continue
            bbox = self._bbox_from_mask(binm)
            if bbox is None:
                continue
            bbox = self._expand_bbox(bbox, W=W, H=H)
            regions.append((cid, area, bbox))

        if len(regions) == 0:
            return np.asarray([], dtype=np.float32)

        regions.sort(key=lambda x: x[1], reverse=True)
        regions = regions[: self.max_regions_per_image]

        gaps: List[float] = []
        for cid, _, (x0, y0, x1, y1) in regions:
            name = class_names[cid]
            text = id_to_text.get(cid, self.prompt_template.format(name))
            t = self._encode_text(text)
            crop = rgb[y0:y1+1, x0:x1+1, :]
            v = self._encode_image_crop(crop)
            gap = float(1.0 - torch.sum(v * t).item())
            gaps.append(gap)
        return np.asarray(gaps, dtype=np.float32)


class EmpiricalDifficultyMaskClip(DifficultyDimension):
    def __init__(
        self,
        predictor=None,
        model_cfg=None,
        class_names: Optional[List[str]] = None,
        device: str = "cuda",
        default_ignore_index: Optional[int] = 255,
    ):
        super().__init__("empirical_iou_maskclip")
        self.predictor = predictor
        self.model_cfg = model_cfg
        self.class_names = class_names
        self.device = device
        self.default_ignore_index = default_ignore_index
        self._model = None

    def _infer_ignore_index(self, m: np.ndarray, meta: Dict[str, Any]) -> Optional[int]:
        if "ignore_index" in meta and meta["ignore_index"] is not None:
            return int(meta["ignore_index"])
        if self.default_ignore_index is not None:
            return int(self.default_ignore_index)
        vals = np.unique(m)
        if 255 in vals:
            return 255
        if 0 in vals and vals.max() > 170:
            return 0
        return None

    def _ensure_model(self, meta: Dict[str, Any]):
        if self.predictor is not None or self._model is not None:
            return
        if self.model_cfg is None:
            raise ValueError("EmpiricalDifficultyMaskClip requires either predictor or model_cfg.")

        from omegaconf import OmegaConf
        from models import build_model

        cfg = OmegaConf.load(self.model_cfg) if isinstance(self.model_cfg, str) else self.model_cfg
        class_names = meta.get("class_names", None) or self.class_names
        if class_names is None:
            raise ValueError("EmpiricalDifficultyMaskClip requires class_names to build MaskClip.")
        model = build_model(cfg.model, class_names=class_names)
        model.eval()
        model.to(self.device)
        self._model = model

    @torch.no_grad()
    def _predict_mask(self, image: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        if self.predictor is not None:
            pred = self.predictor(image, meta=meta)
            return np.asarray(pred, dtype=np.int32)

        self._ensure_model(meta)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        logits = self._model(tensor)
        if logits.shape[-2:] != image.shape[:2]:
            logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
        return pred

    def _compute_per_class_ious(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        meta: Dict[str, Any],
    ) -> np.ndarray:
        gt = gt_mask.astype(np.int32)
        pred = pred_mask.astype(np.int32)
        ignore_index = self._infer_ignore_index(gt, meta)
        valid = (gt != ignore_index) if ignore_index is not None else np.ones_like(gt, dtype=bool)
        valid_classes = np.unique(gt[valid]).tolist()
        ious: List[float] = []
        for cid in valid_classes:
            cid = int(cid)
            pred_c = (pred == cid) & valid
            gt_c = (gt == cid) & valid
            union = np.logical_or(pred_c, gt_c).sum()
            if union == 0:
                continue
            intersection = np.logical_and(pred_c, gt_c).sum()
            ious.append(float(intersection / (union + 1e-8)))
        return np.asarray(ious, dtype=np.float32)

    def get_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> float:
        if image is None or mask is None:
            return 0.0
        meta = meta or {}
        pred = self._predict_mask(image, meta=meta)
        ious = self._compute_per_class_ious(pred, mask, meta)
        if ious.size == 0:
            return 0.0
        return float(np.mean(ious))

    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        if image is None or mask is None:
            return np.asarray([], dtype=np.float32)
        meta = meta or {}
        pred = self._predict_mask(image, meta=meta)
        return self._compute_per_class_ious(pred, mask, meta)
