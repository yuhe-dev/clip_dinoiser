import cv2
import numpy as np
from typing import Optional
from ..dimensions import QualityDimension

class LaplacianSharpness(QualityDimension):
    def __init__(self):
        super().__init__("laplacian")

    def get_score(self, image: np.ndarray, mask: Optional[np.ndarray] = None, meta=None) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta=None
    ) -> np.ndarray:
        """
        Return raw patch-wise Laplacian sharpness scores.
        """
        meta = meta or {}
        patch_size = int(meta.get("patch_size", 32))
        stride = int(meta.get("stride", 16))

        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        patch_scores = []
        if h < patch_size or w < patch_size:
            patch_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        else:
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = gray[y:y + patch_size, x:x + patch_size]
                    patch_scores.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))

        return np.asarray(patch_scores, dtype=np.float32)

class BoundaryGradientAdherence(QualityDimension):
    """
    边界梯度对齐度 (BGA): 
    衡量 Mask 边界与图像物理边缘(梯度)的重合程度。
    """
    def __init__(self, dilation_iteration: int = 1):
        super().__init__("bga")
        self.dilation_iteration = dilation_iteration

    def get_score(self, image: np.ndarray, mask: Optional[np.ndarray] = None, meta=None) -> float:
        # BGA 必须依赖 Mask 才能计算
        if mask is None:
            return 0.0
            
        # 1. 计算图像的梯度幅值 (Image Gradient Magnitude)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用 Sobel 算子计算 x 和 y 方向的导数
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # 计算幅值：sqrt(dx^2 + dy^2)
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        # 归一化梯度图到 [0, 1] 空间，消除图像亮度的绝对影响
        cv2.normalize(magnitude, magnitude, 0, 1, cv2.NORM_MINMAX)

        # 2. 提取 Mask 的边界 (Mask Boundary)
        # 逻辑：将 Mask 膨胀后减去腐蚀后的结果，剩下的就是边缘线
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=self.dilation_iteration)
        eroded = cv2.erode(mask, kernel, iterations=self.dilation_iteration)
        boundary = cv2.absdiff(dilated, eroded)
        
        # 将边界转为二值掩码 (1 表示边界像素，0 表示其他)
        boundary_mask = (boundary > 0).astype(np.float32)

        # 3. 计算对齐得分
        # 逻辑：计算边界掩码覆盖区域内的平均梯度幅值
        boundary_pixel_count = np.sum(boundary_mask)
        
        if boundary_pixel_count == 0:
            return 0.0
            
        # 计算公式：Score = (边界处梯度总和) / (边界像素总数)
        adherence_score = np.sum(magnitude * boundary_mask) / boundary_pixel_count
        
        return float(adherence_score)

    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta=None
    ) -> np.ndarray:
        if mask is None:
            return np.asarray([], dtype=np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        cv2.normalize(magnitude, magnitude, 0, 1, cv2.NORM_MINMAX)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=self.dilation_iteration)
        eroded = cv2.erode(mask, kernel, iterations=self.dilation_iteration)
        boundary = cv2.absdiff(dilated, eroded)
        boundary_mask = boundary > 0

        if not np.any(boundary_mask):
            return np.asarray([], dtype=np.float32)

        return np.asarray(magnitude[boundary_mask], dtype=np.float32)

class WeakTexturePCANoise(QualityDimension):
    """
    Noise level estimation via weak-texture patches + PCA.

    Intuition:
      - Texture looks like noise in high-frequency space, so first select "weak-texture" patches.
      - For those patches, the smallest PCA eigenvalue approximates noise variance (sigma^2).
    
    Output:
      - Estimated noise standard deviation (sigma) in the same intensity scale as the input image (typically 0~255).
    
    References (method family):
      - "Noise level estimation using weak textured patches of a single noisy image" (Okutomi et al., ICIP 2012)
      - "Image noise level estimation by principal component analysis" (Pyatykh et al., IEEE TIP 2013)
    """

    def __init__(
        self,
        patch_size: int = 8,
        stride: int = 8,
        weak_texture_percentile: float = 10.0,
        max_patches: int = 5000,
        tail_eig_k: int = 1,
        min_patches_for_pca: int = 50,
        eps: float = 1e-12,
    ):
        super().__init__("noise_pca_weak_texture")
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.weak_texture_percentile = float(weak_texture_percentile)
        self.max_patches = int(max_patches)
        self.tail_eig_k = int(tail_eig_k)
        self.min_patches_for_pca = int(min_patches_for_pca)
        self.eps = float(eps)

        if self.patch_size <= 1:
            raise ValueError("patch_size must be > 1.")
        if self.stride <= 0:
            raise ValueError("stride must be > 0.")
        if not (0.0 < self.weak_texture_percentile < 100.0):
            raise ValueError("weak_texture_percentile must be in (0, 100).")
        if self.tail_eig_k <= 0:
            raise ValueError("tail_eig_k must be >= 1.")

    def _iter_patches(self, gray: np.ndarray, mask: Optional[np.ndarray]):
        """
        Yield (patch_vector, texture_score) for valid patches.
        texture_score: mean gradient magnitude within the patch (lower => flatter => better for noise estimation).
        """
        H, W = gray.shape[:2]
        ps, st = self.patch_size, self.stride

        # Compute gradient magnitude for texture scoring
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)

        # Prepare mask (optional) as boolean valid region
        valid_mask = None
        if mask is not None:
            if mask.ndim == 3:
                mask_ = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_ = mask
            # Treat >0 as valid ROI
            valid_mask = (mask_.astype(np.float32) > 0.0)

        patches = []
        scores = []

        # Slide window
        for y in range(0, H - ps + 1, st):
            for x in range(0, W - ps + 1, st):
                if valid_mask is not None:
                    roi = valid_mask[y : y + ps, x : x + ps]
                    # Require most pixels inside ROI
                    if roi.mean() < 0.9:
                        continue

                p = gray[y : y + ps, x : x + ps]
                t = gmag[y : y + ps, x : x + ps].mean()

                patches.append(p.reshape(-1))
                scores.append(float(t))

                if len(patches) >= self.max_patches:
                    return np.asarray(patches, dtype=np.float32), np.asarray(scores, dtype=np.float32)

        return np.asarray(patches, dtype=np.float32), np.asarray(scores, dtype=np.float32)

    @staticmethod
    def _box_blur(gray: np.ndarray) -> np.ndarray:
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
        return (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0

    def _iter_patch_noise_scores(self, gray: np.ndarray, mask: Optional[np.ndarray]):
        H, W = gray.shape[:2]
        ps, st = self.patch_size, self.stride

        valid_mask = None
        if mask is not None:
            if mask.ndim == 3:
                mask_ = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_ = mask
            valid_mask = mask_.astype(np.float32) > 0.0

        denoised = self._box_blur(gray.astype(np.float32))
        residual = gray.astype(np.float32) - denoised
        noise_scores = []

        for y in range(0, H - ps + 1, st):
            for x in range(0, W - ps + 1, st):
                if valid_mask is not None:
                    roi = valid_mask[y : y + ps, x : x + ps]
                    if roi.mean() < 0.9:
                        continue

                patch_residual = residual[y : y + ps, x : x + ps]
                noise_scores.append(float(np.sqrt(np.mean(np.square(patch_residual)))))

                if len(noise_scores) >= self.max_patches:
                    return np.asarray(noise_scores, dtype=np.float32)

        return np.asarray(noise_scores, dtype=np.float32)

    def get_score(self, image: np.ndarray, mask: Optional[np.ndarray] = None, meta=None) -> float:
        """
        Return estimated noise sigma (std).
        """
        if image is None or image.size == 0:
            return 0.0

        # 1) Convert to grayscale float32
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        # 2) Extract patches + texture scores
        X, scores = self._iter_patches(gray, mask)

        if X.shape[0] < self.min_patches_for_pca:
            return 0.0  # not enough evidence

        # 3) Select weak-texture patches
        thr = np.percentile(scores, self.weak_texture_percentile)
        idx = np.where(scores <= thr)[0]
        if idx.size < self.min_patches_for_pca:
            # fallback: take the weakest min_patches_for_pca
            idx = np.argsort(scores)[: self.min_patches_for_pca]

        Xw = X[idx]  # shape: (N, D), D = patch_size^2
        if Xw.shape[0] < self.min_patches_for_pca:
            return 0.0

        # 4) PCA on weak-texture patches
        # Center features
        Xw = Xw - Xw.mean(axis=0, keepdims=True)

        # Covariance: D x D (D is small, e.g., 64 for 8x8)
        N = Xw.shape[0]
        C = (Xw.T @ Xw) / max(N - 1, 1)

        # Eigenvalues (ascending for eigvalsh)
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.clip(eigvals, 0.0, None)

        # 5) Estimate noise variance from smallest tail_eig_k eigenvalues
        k = min(self.tail_eig_k, eigvals.size)
        sigma2 = float(np.mean(eigvals[:k]))
        sigma = float(np.sqrt(max(sigma2, self.eps)))

        return sigma

    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta=None
    ) -> np.ndarray:
        if image is None or image.size == 0:
            return np.asarray([], dtype=np.float32)

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        values = self._iter_patch_noise_scores(gray, mask)
        return np.asarray(values, dtype=np.float32)
