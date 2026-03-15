from __future__ import annotations

import json
import os
import shutil

import numpy as np

from .projector import SliceFeatureProjector


class SliceReportExporter:
    UMAP_RANDOM_STATE = 42
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1

    def export(
        self,
        projected_dir: str,
        cluster_dir: str,
        output_dir: str,
        image_root: str | None = None,
    ) -> None:
        projected = SliceFeatureProjector.load(projected_dir)
        cluster_payload = np.load(os.path.join(cluster_dir, "slice_result.npz"), allow_pickle=True)
        with open(os.path.join(cluster_dir, "slice_result_meta.json"), "r", encoding="utf-8") as f:
            cluster_meta = json.load(f)

        sample_ids = [str(sample_id) for sample_id in cluster_payload["sample_ids"].tolist()]
        membership = np.asarray(cluster_payload["membership"], dtype=np.float32)
        hard_assignment = np.asarray(cluster_payload["hard_assignment"], dtype=np.int64)
        slice_weights = np.asarray(cluster_payload["slice_weights"], dtype=np.float32)
        centers = np.asarray(cluster_payload["centers"], dtype=np.float32)
        embedding_2d = self._compute_umap_2d(projected.matrix)
        slice_centers_2d = self._compute_slice_centers_2d(
            embedding_2d=embedding_2d,
            membership=membership,
            slice_weights=slice_weights,
            hard_assignment=hard_assignment,
        )

        os.makedirs(output_dir, exist_ok=True)

        run_summary = {
            "run_id": os.path.basename(os.path.abspath(output_dir)),
            "finder": cluster_meta["finder"],
            "num_slices": int(cluster_meta["num_slices"]),
            "sample_count": int(cluster_meta["sample_count"]),
            "block_order": list(projected.block_ranges.keys()),
            "block_ranges": {name: [start, end] for name, (start, end) in projected.block_ranges.items()},
            "slice_weights": slice_weights.astype(float).tolist(),
            "hard_assignment_counts": np.bincount(
                hard_assignment,
                minlength=int(cluster_meta["num_slices"]),
            ).astype(int).tolist(),
            "membership_health": {
                "row_sum_min": float(membership.sum(axis=1).min()),
                "row_sum_max": float(membership.sum(axis=1).max()),
                "avg_max_membership": float(membership.max(axis=1).mean()),
                "avg_entropy": float((-(membership * np.log(np.clip(membership, 1e-12, None))).sum(axis=1)).mean()),
            },
            "embedding": {
                "method": "umap",
                "random_state": int(self.UMAP_RANDOM_STATE),
                "n_neighbors": int(self.UMAP_N_NEIGHBORS),
                "min_dist": float(self.UMAP_MIN_DIST),
            },
        }

        slices = []
        for slice_id in range(int(cluster_meta["num_slices"])):
            mask = hard_assignment == slice_id
            max_membership = membership[:, slice_id]
            slices.append(
                {
                    "slice_id": f"slice_{slice_id:02d}",
                    "index": int(slice_id),
                    "weight": float(slice_weights[slice_id]),
                    "hard_count": int(mask.sum()),
                    "avg_max_membership": float(max_membership.mean()),
                    "center": centers[slice_id].astype(float).tolist(),
                }
            )

        selected_sample_ids = set()

        feature_schema = {
            "block_order": list(projected.block_ranges.keys()),
            "blocks": {
                name: {
                    "start": int(start),
                    "end": int(end),
                    "width": int(end - start),
                }
                for name, (start, end) in projected.block_ranges.items()
            },
        }

        entropy = -(membership * np.log(np.clip(membership, 1e-12, None))).sum(axis=1)
        global_mean = projected.matrix.mean(axis=0)

        enriched_slices = []
        for slice_info in slices:
            slice_index = int(slice_info["index"])
            sample_weights = membership[:, slice_index]
            weighted_mean = (sample_weights[:, None] * projected.matrix).sum(axis=0) / np.clip(sample_weights.sum(), 1e-12, None)
            deltas = weighted_mean - global_mean

            block_portrait = []
            for block_name, (start, end) in projected.block_ranges.items():
                block_portrait.append(
                    {
                        "block": block_name,
                        "slice_mean": float(weighted_mean[start:end].mean()),
                        "global_mean": float(global_mean[start:end].mean()),
                        "delta": float(deltas[start:end].mean()),
                    }
                )

            top_shift_indices = np.argsort(-np.abs(deltas))[: min(10, deltas.shape[0])]
            top_shifted_features = []
            for feature_index in top_shift_indices.tolist():
                block_name = self._find_block_name(projected.block_ranges, feature_index)
                top_shifted_features.append(
                    {
                        "feature_index": int(feature_index),
                        "block": block_name,
                        "field": f"dim_{feature_index:03d}",
                        "slice_mean": float(weighted_mean[feature_index]),
                        "global_mean": float(global_mean[feature_index]),
                        "shift_score": float(deltas[feature_index]),
                    }
                )

            representative_samples = self._top_sample_ids(sample_ids, sample_weights, descending=True)
            center_scores = -np.sum((projected.matrix - centers[slice_index][None, :]) ** 2, axis=1)
            center_samples = self._top_sample_ids(sample_ids, center_scores, descending=True)
            ambiguous_mask = hard_assignment == slice_index
            ambiguous_scores = np.where(ambiguous_mask, entropy, -np.inf)
            ambiguous_samples = self._top_sample_ids(sample_ids, ambiguous_scores, descending=True)
            selected_sample_ids.update(representative_samples)
            selected_sample_ids.update(center_samples)
            selected_sample_ids.update(ambiguous_samples)

            slice_info = dict(slice_info)
            slice_info["avg_entropy"] = float((sample_weights * entropy).sum() / np.clip(sample_weights.sum(), 1e-12, None))
            slice_info["block_portrait"] = block_portrait
            slice_info["top_shifted_features"] = top_shifted_features
            slice_info["representative_samples"] = representative_samples
            slice_info["center_samples"] = center_samples
            slice_info["ambiguous_samples"] = ambiguous_samples
            enriched_slices.append(slice_info)

        samples = []
        embedding_payload = []
        for index, sample_id in enumerate(sample_ids):
            order = np.argsort(-membership[index]).astype(int).tolist()
            image_url = ""
            if image_root is None:
                image_url = sample_id
            elif sample_id in selected_sample_ids:
                image_url = self._copy_sample_image(
                    image_root=image_root,
                    output_dir=output_dir,
                    sample_id=sample_id,
                )
            samples.append(
                {
                    "sample_id": sample_id,
                    "image_rel": sample_id,
                    "image_url": image_url,
                    "hard_assignment": int(hard_assignment[index]),
                    "max_membership": float(membership[index].max()),
                    "membership_vector": membership[index].astype(float).tolist(),
                    "slice_rankings": order,
                }
            )
            embedding_payload.append(
                {
                    "sample_id": sample_id,
                    "x": float(embedding_2d[index, 0]),
                    "y": float(embedding_2d[index, 1]),
                    "hard_assignment": int(hard_assignment[index]),
                    "max_membership": float(membership[index].max()),
                }
            )

        self._write_json(os.path.join(output_dir, "run_summary.json"), run_summary)
        self._write_json(os.path.join(output_dir, "slices.json"), enriched_slices)
        self._write_json(os.path.join(output_dir, "samples.json"), samples)
        self._write_json(os.path.join(output_dir, "feature_schema.json"), feature_schema)
        self._write_json(os.path.join(output_dir, "embedding_2d.json"), embedding_payload)
        self._write_json(os.path.join(output_dir, "slice_centers_2d.json"), slice_centers_2d)

    @staticmethod
    def _write_json(path: str, payload: object) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _find_block_name(block_ranges: dict[str, tuple[int, int]], feature_index: int) -> str:
        for block_name, (start, end) in block_ranges.items():
            if start <= feature_index < end:
                return block_name
        return "unknown"

    @staticmethod
    def _top_sample_ids(sample_ids: list[str], scores: np.ndarray, descending: bool, top_k: int = 12) -> list[str]:
        order = np.argsort(-scores if descending else scores)
        sample_list = []
        for index in order.tolist():
            if not np.isfinite(scores[index]):
                continue
            sample_list.append(sample_ids[index])
            if len(sample_list) >= top_k:
                break
        return sample_list

    @staticmethod
    def _copy_sample_image(image_root: str, output_dir: str, sample_id: str) -> str:
        src_path = os.path.join(image_root, sample_id)
        thumbnails_dir = os.path.join(output_dir, "thumbnails")
        dst_path = os.path.join(thumbnails_dir, sample_id)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
        return f"./thumbnails/{sample_id}"

    def _compute_umap_2d(self, matrix: np.ndarray) -> np.ndarray:
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError(
                "UMAP export requires the 'umap-learn' package to be installed in the Python environment."
            ) from exc

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.UMAP_N_NEIGHBORS,
            min_dist=self.UMAP_MIN_DIST,
            random_state=self.UMAP_RANDOM_STATE,
        )
        embedding = np.asarray(reducer.fit_transform(matrix), dtype=np.float32)
        if embedding.shape != (matrix.shape[0], 2):
            raise ValueError(f"Expected UMAP embedding with shape {(matrix.shape[0], 2)}, got {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise ValueError("UMAP embedding contains non-finite coordinates.")
        return embedding

    @staticmethod
    def _compute_slice_centers_2d(
        embedding_2d: np.ndarray,
        membership: np.ndarray,
        slice_weights: np.ndarray,
        hard_assignment: np.ndarray,
    ) -> list[dict[str, float | int | str]]:
        centers = []
        num_slices = membership.shape[1]
        for slice_index in range(num_slices):
            weights = membership[:, slice_index]
            norm = float(np.clip(weights.sum(), 1e-12, None))
            center_xy = (weights[:, None] * embedding_2d).sum(axis=0) / norm
            if not np.isfinite(center_xy).all():
                raise ValueError(f"Non-finite 2D center computed for slice {slice_index}.")
            centers.append(
                {
                    "slice_id": f"slice_{slice_index:02d}",
                    "x": float(center_xy[0]),
                    "y": float(center_xy[1]),
                    "weight": float(slice_weights[slice_index]),
                    "hard_count": int((hard_assignment == slice_index).sum()),
                }
            )
        return centers
