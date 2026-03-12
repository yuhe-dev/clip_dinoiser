import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROW_SPECS: List[Tuple[str, str, str]] = [
    ("quality", "laplacian", "hist"),
    ("quality", "noise_pca", "hist"),
    ("quality", "bga", "hist"),
    ("difficulty", "small_ratio", "delta_profile"),
    ("difficulty", "visual_semantic_gap", "hist"),
    ("difficulty", "empirical_iou", "hist"),
    ("coverage", "knn_local_density", "profile"),
    ("coverage", "prototype_distance", "profile"),
]


def load_processed_records(processed_path: str) -> List[Dict[str, object]]:
    arr = np.load(processed_path, allow_pickle=True)
    return [dict(item) for item in arr.tolist()]


def index_records_by_image_rel(records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(record["image_rel"]): dict(record) for record in records}


def resample_sequence(sequence: np.ndarray, target_width: int = 16) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.size == 0:
        return np.zeros((target_width,), dtype=np.float32)
    if seq.size == target_width:
        return seq.astype(np.float32)
    if seq.size == 1:
        return np.repeat(seq.astype(np.float32), target_width)
    x_old = np.linspace(0.0, 1.0, seq.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_width, dtype=np.float32)
    return np.interp(x_new, x_old, seq).astype(np.float32)


def pool_profile_sequence(sequence: np.ndarray, target_width: int = 16) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.size == 0:
        return np.zeros((target_width,), dtype=np.float32)
    if seq.size <= target_width:
        return resample_sequence(seq, target_width=target_width)

    chunks = np.array_split(seq, target_width)
    pooled = []
    for chunk in chunks:
        if chunk.size == 0:
            pooled.append(0.0)
        else:
            pooled.append(float(chunk.mean()))
    return np.asarray(pooled, dtype=np.float32)


def _extract_row_sequence(record: Dict[str, object], feature_name: str, field_name: str) -> np.ndarray:
    features = dict(record["features"])
    block = dict(features[feature_name])
    return np.asarray(block.get(field_name, np.asarray([], dtype=np.float32)), dtype=np.float32)


def assemble_feature_landscape_matrix(
    quality_record: Dict[str, object],
    difficulty_record: Dict[str, object],
    coverage_record: Dict[str, object],
    target_width: int = 16,
) -> Tuple[np.ndarray, Dict[str, object]]:
    record_map = {
        "quality": quality_record,
        "difficulty": difficulty_record,
        "coverage": coverage_record,
    }

    rows: List[np.ndarray] = []
    row_labels: List[str] = []
    for dimension_name, feature_name, field_name in ROW_SPECS:
        seq = _extract_row_sequence(record_map[dimension_name], feature_name, field_name)
        if dimension_name == "coverage":
            vis_seq = pool_profile_sequence(seq, target_width=target_width)
        else:
            vis_seq = resample_sequence(seq, target_width=target_width)
        rows.append(vis_seq.astype(np.float32))
        row_labels.append(f"{dimension_name}.{feature_name}")

    matrix = np.vstack(rows).astype(np.float32)
    meta = {
        "image_rel": str(quality_record["image_rel"]),
        "annotation_rel": str(quality_record.get("annotation_rel", "")),
        "row_labels": row_labels,
        "group_slices": {
            "quality": [0, 3],
            "difficulty": [3, 6],
            "coverage": [6, 8],
        },
        "target_width": int(target_width),
    }
    return matrix, meta


def export_landscape_matrix_json(output_path: str, matrix: np.ndarray, meta: Dict[str, object]) -> None:
    payload = {
        "image_rel": meta["image_rel"],
        "annotation_rel": meta.get("annotation_rel", ""),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "row_labels": list(meta["row_labels"]),
        "group_slices": dict(meta["group_slices"]),
        "target_width": int(meta["target_width"]),
        "matrix": np.asarray(matrix, dtype=np.float32).tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def render_feature_landscape_3d(output_path: str, matrix: np.ndarray, meta: Dict[str, object], mode: str = "paper") -> None:
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = np.arange(matrix.shape[1], dtype=np.float32)
    cmap_by_group = {
        "quality": matplotlib.colormaps["OrRd"],
        "difficulty": matplotlib.colormaps["YlGnBu"],
        "coverage": matplotlib.colormaps["PuBu"],
    }

    for row_idx in range(matrix.shape[0]):
        y = np.full_like(x, row_idx, dtype=np.float32)
        z = matrix[row_idx]
        group_name = "quality" if row_idx < 3 else "difficulty" if row_idx < 6 else "coverage"
        color_values = cmap_by_group[group_name](np.linspace(0.35, 0.85, matrix.shape[1]))
        ax.plot(x, y, z, color=color_values[-1], linewidth=2.0)
        ax.plot_trisurf(
            np.concatenate([x, x]),
            np.concatenate([y, y + 0.85]),
            np.concatenate([np.zeros_like(z), z]),
            color=color_values[-1],
            alpha=0.55 if mode == "paper" else 0.7,
            linewidth=0.0,
            shade=True,
        )

    ax.set_xlabel("Visualized Feature Position")
    ax.set_ylabel("Feature Row")
    ax.set_zlabel("Normalized Value")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(meta["row_labels"], fontsize=8)
    ax.set_title("Unified Feature Landscape")
    ax.view_init(elev=28, azim=-58)
    if mode == "paper":
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_feature_landscape_ribbon(output_path: str, matrix: np.ndarray, meta: Dict[str, object], mode: str = "paper") -> None:
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = np.arange(matrix.shape[1], dtype=np.float32)
    ribbon_width = 0.72
    gap_positions = {
        "quality": [0.0, 1.1, 2.2],
        "difficulty": [4.9, 6.0, 7.1],
        "coverage": [10.3, 11.5],
    }
    ribbon_positions = (
        gap_positions["quality"]
        + gap_positions["difficulty"]
        + gap_positions["coverage"]
    )
    color_by_group = {
        "quality": "#E67E22",
        "difficulty": "#0E9F8A",
        "coverage": "#2F5AA8",
    }
    label_added = {key: False for key in color_by_group}

    for row_idx in range(matrix.shape[0]):
        z = matrix[row_idx]
        group_name = "quality" if row_idx < 3 else "difficulty" if row_idx < 6 else "coverage"
        base_y = float(ribbon_positions[row_idx])
        y_left = np.full_like(x, base_y - ribbon_width / 2.0, dtype=np.float32)
        y_right = np.full_like(x, base_y + ribbon_width / 2.0, dtype=np.float32)
        band_color = color_by_group[group_name]
        legend_label = group_name.capitalize() if not label_added[group_name] else None
        label_added[group_name] = True

        X = np.vstack([x, x])
        Y = np.vstack([y_left, y_right])
        Z = np.vstack([z, z])
        ax.plot_surface(
            X,
            Y,
            Z,
            color=band_color,
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=0.9 if mode == "paper" else 0.95,
        )
        ax.plot(
            x,
            np.full_like(x, base_y, dtype=np.float32),
            z,
            color=band_color,
            linewidth=2.2,
            label=legend_label,
        )

        if mode != "paper":
            ax.plot(
                x,
                np.full_like(x, base_y, dtype=np.float32),
                np.zeros_like(z),
                color=(0.35, 0.35, 0.35, 0.35),
                linewidth=0.8,
                linestyle="--",
            )

    ax.set_xlabel("Feature Position")
    ax.set_ylabel("")
    ax.set_zlabel("Normalized Value")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title("Unified Feature Ribbon Landscape")
    ax.view_init(elev=24, azim=-66)
    ax.set_xlim(0, matrix.shape[1] - 1)
    ax.set_ylim(-0.7, max(ribbon_positions) + 0.9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(axis="x", which="major", labelsize=9, pad=2)
    ax.tick_params(axis="z", which="major", labelsize=9, pad=2)
    if mode == "paper":
        ax.grid(False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_feature_landscape_heatmap(output_path: str, matrix: np.ndarray, meta: Dict[str, object], mode: str = "analysis") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(meta["row_labels"], fontsize=8)
    ax.set_xlabel("Visualized Feature Position")
    ax.set_title("Unified Feature Landscape Heatmap")
    if mode != "paper":
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pick_image_rel(requested_image_rel: str, indexed_bundles: Sequence[Dict[str, Dict[str, object]]]) -> str:
    if requested_image_rel:
        return requested_image_rel
    common = set(indexed_bundles[0].keys())
    for bundle in indexed_bundles[1:]:
        common &= set(bundle.keys())
    if not common:
        raise ValueError("No shared image_rel found across quality, difficulty, and coverage bundles.")
    return sorted(common)[0]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a 3D feature landscape for one processed sample.")
    parser.add_argument("--image-rel", default=None)
    parser.add_argument("--quality-path", default="./data/data_feature/quality/quality_processed_features.npy")
    parser.add_argument("--difficulty-path", default="./data/data_feature/difficulty/difficulty_processed_features.npy")
    parser.add_argument("--coverage-path", default="./data/data_feature/coverage/coverage_processed_features.npy")
    parser.add_argument("--output-dir", default="./figures/feature_landscape")
    parser.add_argument("--mode", choices=["paper", "analysis"], default="paper")
    parser.add_argument("--style", choices=["ribbon", "surface"], default="ribbon")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--save-matrix-json", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    quality_records = index_records_by_image_rel(load_processed_records(os.path.abspath(args.quality_path)))
    difficulty_records = index_records_by_image_rel(load_processed_records(os.path.abspath(args.difficulty_path)))
    coverage_records = index_records_by_image_rel(load_processed_records(os.path.abspath(args.coverage_path)))
    image_rel = _pick_image_rel(args.image_rel, [quality_records, difficulty_records, coverage_records])

    matrix, meta = assemble_feature_landscape_matrix(
        quality_record=quality_records[image_rel],
        difficulty_record=difficulty_records[image_rel],
        coverage_record=coverage_records[image_rel],
        target_width=int(args.width),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    sample_stem = os.path.splitext(os.path.basename(image_rel))[0]
    png_3d = os.path.join(args.output_dir, f"{sample_stem}_feature_landscape_3d.png")
    png_heatmap = os.path.join(args.output_dir, f"{sample_stem}_feature_landscape_heatmap.png")
    if args.style == "surface":
        render_feature_landscape_3d(png_3d, matrix, meta, mode=args.mode)
    else:
        render_feature_landscape_ribbon(png_3d, matrix, meta, mode=args.mode)
    render_feature_landscape_heatmap(png_heatmap, matrix, meta, mode=args.mode)

    if args.save_matrix_json or True:
        json_path = os.path.join(args.output_dir, f"{sample_stem}_feature_landscape_matrix.json")
        export_landscape_matrix_json(json_path, matrix, meta)
        print(f"[feature-landscape] wrote {json_path}")

    print(f"[feature-landscape] wrote {png_3d}")
    print(f"[feature-landscape] wrote {png_heatmap}")


if __name__ == "__main__":
    main()
