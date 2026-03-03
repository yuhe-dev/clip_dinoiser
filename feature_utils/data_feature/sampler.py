import os
import cv2
import random
import numpy as np
import torch
from tqdm import tqdm
from .base import BaseMetric

class StratifiedSampler:
    """
    通用分层采样器：
    1. 接受任何 BaseMetric 的子类作为评估标准。
    2. 实现基于类别分布对齐的采样逻辑。
    3. 自动生成 Low, High, Mixed 三个控制变量子集。
    """
    def __init__(self, metric: BaseMetric, num_samples_per_tier=1000, logger=None):
        self.metric = metric
        self.target = num_samples_per_tier
        self.logger = logger
        self.img_scores = {}
        self.class_buckets = {i: [] for i in range(80)} # COCO-Stuff 171类

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[Sampler] {msg}")

    def get_subsets_with_scores(self, pool_paths, ann_dir):
        """
        执行完整的采样流程
        :param pool_paths: 初始随机选取的图片路径列表 (建议 20000+)
        :param ann_dir: 存放 _labelTrainIds.png 标注的目录
        :return: low_paths, high_paths, mixed_paths
        """
        
        # # --- Step 1: 扫描池子并计算得分 ---
        # self._log(f"Step 1: Calculating {self.metric.name} scores for {len(pool_paths)} images...")
        # for img_path in tqdm(pool_paths, desc=f"Evaluating {self.metric.name}"):
        #     # 读取图像和 Mask (如果指标需要)
        #     image = cv2.imread(img_path)
        #     if image is None: continue

        #     stem = os.path.splitext(os.path.basename(img_path))[0]
        #     # mask_name = os.path.basename(img_path).replace('.jpg', '_labelTrainIds.png')
        #     mask_name = stem + "_labelTrainIds.png"
        #     mask_path = os.path.join(ann_dir, mask_name)
        #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
        #     # 多态调用：无论是什么指标，接口统一
        #     meta = None
        #     if hasattr(self.metric, "dim_type") and self.metric.dim_type == "Difficulty":
        #         meta = {
        #             "ignore_index": 255,
        #             "thing_ids": list(range(0, 80))
        #         }
        #     score = self.metric.get_score(image, mask, meta=meta)
        #     self.img_scores[img_path] = score
            
        #     # 建立类别索引 (用于分布对齐)
        #     if mask is not None:
        #         THING_IDS = set(range(0, 80))
        #         IGNORE = 255

        #         unique_ids = np.unique(mask)
        #         for uid in unique_ids:
        #             if uid in THING_IDS: # 过滤 ignore_label
        #                 self.class_buckets[uid].append(img_path)

        # --- Step 1: 扫描池子并计算得分 ---
        self._log(f"Step 1: Calculating {self.metric.name} scores for {len(pool_paths)} images...")

        needs_image = getattr(self.metric, "needs_image", True)
        needs_mask  = getattr(self.metric, "needs_mask", True)

        for img_path in tqdm(pool_paths, desc=f"Evaluating {self.metric.name}"):

            # 1) 先准备 meta（LocalDensity 就靠这个）
            meta = {"img_path": img_path, "path": img_path}  # 两个键都给，兼容你写的 lookup

            # difficulty 指标需要的 meta（你原来那段保留）
            if getattr(self.metric, "dim_type", None) == "Difficulty":
                meta.update({
                    "ignore_index": 255,
                    "thing_ids": list(range(0, 80)),
                })

            # 2) 按需读取 image/mask（LocalDensity 会跳过）
            image = None
            if needs_image:
                image = cv2.imread(img_path)
                if image is None:
                    continue

            mask = None
            if needs_mask:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                mask_name = stem + "_labelTrainIds.png"
                mask_path = os.path.join(ann_dir, mask_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

            # 3) 多态调用（LocalDensity 只会用 meta）
            score = self.metric.get_score(image, mask, meta=meta)
            self.img_scores[img_path] = score

            # 4) 建桶：只有你需要“类别分布对齐”时才必须 mask
            #    对 LocalDensity 这种 coverage 指标，如果你仍然想按 thing 类对齐，那就 needs_mask=True
            if mask is not None:
                STUFF_IDS = set(range(0, 80))
                unique_ids = np.unique(mask)
                for uid in unique_ids:
                    if int(uid) in STUFF_IDS:
                        self.class_buckets[int(uid)].append(img_path)

        # --- Step 2: 计算各类别配额 (Quota) ---
        self._log("Step 2: Calculating class quotas for distribution alignment...")
        total_occurrences = sum(len(imgs) for imgs in self.class_buckets.values())
        class_quotas = {}
        for uid, images in self.class_buckets.items():
            if len(images) == 0: continue
            proportion = len(images) / total_occurrences
            # 乘以 4 倍系数以应对多标签去重导致的损耗
            class_quotas[uid] = max(1, int(proportion * self.target * 4))

        # --- Step 3: 在类别桶内采样 ---
        self._log("Step 3: Sampling Low, High, and Mixed candidates from buckets...")
        low_set, high_set, mixed_set = set(), set(), set()

        for uid, images in self.class_buckets.items():
            if len(images) == 0: continue
            
            # 按分数排序 (多态的核心体现：排序依据由传入的 metric 决定)
            images.sort(key=lambda p: self.img_scores[p])
            n = len(images)
            q = min(class_quotas[uid], n)

            # A. Low Subset: 取分值最低的
            for i in range(q):
                low_set.add(images[i])

            # B. High Subset: 取分值最高的
            for i in range(q):
                high_set.add(images[n - 1 - i])

            # C. Mixed Subset: 模拟混合逻辑 (取桶内分布的过渡带)
            # 取中间一段 (n//4 到 n//4+q/2) 和 (3n//4-q/2 到 3n//4)
            mid_low_idx = n // 4
            mid_high_idx = 3 * n // 4
            half_q = max(1, q // 2)
            for i in range(min(half_q, n - mid_low_idx)):
                mixed_set.add(images[mid_low_idx + i])
            for i in range(min(half_q, mid_high_idx)):
                mixed_set.add(images[mid_high_idx - i])

        # --- Step 4: 最终对齐与截断 ---
        self._log("Step 4: Finalizing and equalizing subset sizes...")
        low_list = self._finalize(low_set, pool_paths)
        high_list = self._finalize(high_set, pool_paths)
        mixed_list = self._finalize(mixed_set, pool_paths)
        rand_list = random.sample(pool_paths, len(mixed_list))
        
        return low_list, high_list, mixed_list, rand_list, self.img_scores

    def _finalize(self, path_set, pool_paths):
        """确保最终生成的列表长度严格等于目标值，并进行去重/补齐"""
        p_list = list(path_set)
        if len(p_list) > self.target:
            return random.sample(p_list, self.target)
        elif len(p_list) < self.target:
            self._log(f"Warning: Set undersized ({len(p_list)}). Padding from original pool...")
            remaining = list(set(pool_paths) - path_set)
            padding = random.sample(remaining, self.target - len(p_list))
            return p_list + padding
        return p_list