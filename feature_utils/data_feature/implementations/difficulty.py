import cv2
import numpy as np
from typing import Optional, Dict, Any, Iterable, List
from ..dimensions import DifficultyDimension
import torch


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
    ):
        super().__init__("small_object_ratio")
        self.tau_ratio = float(tau_ratio)
        self.connectivity = int(connectivity)
        self.thing_id_start = int(thing_id_start)
        self.num_things = int(num_things)
        self.default_ignore_index = default_ignore_index
        self.use_things_only = bool(use_things_only)

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
        # Compatibility vectorization; multi-threshold vectorization can replace this later.
        return np.asarray([self.get_score(image, mask, meta=meta)], dtype=np.float32)
    
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
            tokens = self.tokenizer([text]).to(self.device) if self.tokenizer is not None else text
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
        # ---- You must adapt this block to your CLIP implementation ----
        # Expected: return a normalized embedding vector [D]
        # crop_rgb is HxWx3 in RGB uint8
        if hasattr(self.clip_model, "encode_image"):
            # open_clip style: needs preprocessing -> tensor
            # You likely already have a preprocess transform in your project.
            raise RuntimeError("Please adapt _encode_image_crop() with your preprocess.")
        else:
            raise RuntimeError("Please adapt _encode_image_crop() to your CLIP implementation.")

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
        # Compatibility vectorization; region-gap histogram encoding can replace this later.
        return np.asarray([self.get_score(image, mask, meta=meta)], dtype=np.float32)
