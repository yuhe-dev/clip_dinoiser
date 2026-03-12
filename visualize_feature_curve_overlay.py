import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_feature_landscape import (
    assemble_feature_landscape_matrix,
    index_records_by_image_rel,
    load_processed_records,
)


BASE_GROUP_COLORS = {
    "quality": "#E67E22",
    "difficulty": "#0E9F8A",
    "coverage": "#2F5AA8",
}


def _hex_to_rgb01(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.asarray([int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _rgb01_to_hex(rgb: np.ndarray) -> str:
    vals = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
    return "#{:02X}{:02X}{:02X}".format(int(vals[0]), int(vals[1]), int(vals[2]))


def _mix_with_white(hex_color: str, keep_ratio: float) -> str:
    rgb = _hex_to_rgb01(hex_color)
    white = np.ones(3, dtype=np.float32)
    mixed = white * (1.0 - keep_ratio) + rgb * keep_ratio
    return _rgb01_to_hex(mixed)


def build_curve_style_metadata(row_labels: List[str]) -> Dict[str, object]:
    curve_labels = []
    curve_colors = []
    group_counts = {"quality": 0, "difficulty": 0, "coverage": 0}
    group_totals = {
        "quality": sum(1 for label in row_labels if label.startswith("quality.")),
        "difficulty": sum(1 for label in row_labels if label.startswith("difficulty.")),
        "coverage": sum(1 for label in row_labels if label.startswith("coverage.")),
    }

    for label in row_labels:
        group_name, feature_name = label.split(".", 1)
        group_counts[group_name] += 1
        total = max(group_totals[group_name], 1)
        position = group_counts[group_name] - 1
        if total == 1:
            keep_ratio = 0.85
        else:
            keep_ratio = 0.45 + 0.4 * (position / (total - 1))
        curve_colors.append(_mix_with_white(BASE_GROUP_COLORS[group_name], keep_ratio))
        curve_labels.append(feature_name.replace("_", " ").title())

    return {
        "curve_labels": curve_labels,
        "curve_colors": curve_colors,
        "legend_labels": {
            "quality": "Quality",
            "difficulty": "Difficulty",
            "coverage": "Coverage",
        },
    }


def export_curve_overlay_json(output_path: str, matrix: np.ndarray, meta: Dict[str, object], style_meta: Dict[str, object]) -> None:
    payload = {
        "image_rel": meta["image_rel"],
        "annotation_rel": meta.get("annotation_rel", ""),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "row_labels": list(meta["row_labels"]),
        "curve_labels": list(style_meta["curve_labels"]),
        "curve_colors": list(style_meta["curve_colors"]),
        "legend_labels": dict(style_meta["legend_labels"]),
        "target_width": int(meta["target_width"]),
        "matrix": np.asarray(matrix, dtype=np.float32).tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_ridge_layout(
    matrix: np.ndarray,
    row_labels: List[str],
    intra_gap: float = 0.9,
    inter_gap: float = 1.8,
) -> Dict[str, object]:
    baselines = []
    current = 0.0
    last_group = None
    group_rows: Dict[str, List[int]] = {"quality": [], "difficulty": [], "coverage": []}

    for idx, label in enumerate(row_labels):
        group_name = label.split(".", 1)[0]
        if last_group is None:
            current = 0.0
        elif group_name == last_group:
            current += intra_gap
        else:
            current += inter_gap
        baselines.append(current)
        group_rows[group_name].append(idx)
        last_group = group_name

    baselines_arr = np.asarray(baselines, dtype=np.float32)
    group_centers = {
        group_name: float(np.mean([baselines_arr[idx] for idx in indices]))
        for group_name, indices in group_rows.items()
        if indices
    }
    return {
        "baselines": baselines_arr,
        "group_centers": group_centers,
    }


def render_curve_overlay(output_path: str, matrix: np.ndarray, meta: Dict[str, object], style_meta: Dict[str, object], mode: str = "paper") -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(matrix.shape[1], dtype=np.float32)

    for row_idx, label in enumerate(meta["row_labels"]):
        y = matrix[row_idx]
        curve_label = style_meta["curve_labels"][row_idx]
        curve_color = style_meta["curve_colors"][row_idx]
        ax.plot(
            x,
            y,
            color=curve_color,
            linewidth=2.6 if mode == "paper" else 2.0,
            alpha=0.95,
            label=curve_label,
        )

    ax.set_xlim(0, matrix.shape[1] - 1)
    ax.set_xlabel("Visualized Feature Position")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Unified Feature Curve Overlay")

    if mode == "paper":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    else:
        ax.grid(alpha=0.25, linewidth=0.8)

    handles = []
    for group_name, legend_label in style_meta["legend_labels"].items():
        handles.append(plt.Line2D([0], [0], color=BASE_GROUP_COLORS[group_name], lw=3, label=legend_label))
    legend1 = ax.legend(handles=handles, loc="upper left", frameon=False, title="Dimension")
    ax.add_artist(legend1)

    curve_handles = []
    for curve_label, curve_color in zip(style_meta["curve_labels"], style_meta["curve_colors"]):
        curve_handles.append(plt.Line2D([0], [0], color=curve_color, lw=2, label=curve_label))
    ax.legend(
        handles=curve_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Feature",
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_curve_ridges(output_path: str, matrix: np.ndarray, meta: Dict[str, object], style_meta: Dict[str, object], mode: str = "paper") -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.8))
    x = np.arange(matrix.shape[1], dtype=np.float32)
    layout = build_ridge_layout(matrix, list(meta["row_labels"]))
    baselines = layout["baselines"]
    group_centers = layout["group_centers"]

    for row_idx, label in enumerate(meta["row_labels"]):
        y = matrix[row_idx]
        base = baselines[row_idx]
        curve_color = style_meta["curve_colors"][row_idx]
        ax.fill_between(
            x,
            base,
            base + y,
            color=curve_color,
            alpha=0.28 if mode == "paper" else 0.36,
            linewidth=0,
        )
        ax.plot(
            x,
            base + y,
            color=curve_color,
            linewidth=2.4 if mode == "paper" else 2.0,
            alpha=0.98,
        )
        ax.hlines(base, x[0], x[-1], colors=(0.2, 0.2, 0.2, 0.10), linewidth=0.8)

    ax.set_xlim(0, matrix.shape[1] - 1)
    ax.set_xlabel("Feature Position")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("Unified Feature Ridge Overlay")

    if mode == "paper":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", alpha=0.08, linewidth=0.7)
    else:
        ax.grid(alpha=0.18, linewidth=0.8)

    for group_name, center_y in group_centers.items():
        ax.text(
            -1.8,
            center_y,
            style_meta["legend_labels"][group_name],
            color=BASE_GROUP_COLORS[group_name],
            fontsize=11 if mode == "paper" else 10,
            fontweight="bold",
            ha="right",
            va="center",
            clip_on=False,
        )

    curve_handles = []
    for curve_label, curve_color in zip(style_meta["curve_labels"], style_meta["curve_colors"]):
        curve_handles.append(plt.Line2D([0], [0], color=curve_color, lw=2, label=curve_label))
    feature_legend = ax.legend(
        handles=curve_handles,
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        frameon=False,
        title="Feature",
        ncol=1,
    )
    plt.setp(feature_legend.get_title(), fontsize=11, fontweight="bold")
    for text in feature_legend.get_texts():
        text.set_fontsize(9)

    dimension_handles = []
    for group_name, legend_label in style_meta["legend_labels"].items():
        dimension_handles.append(plt.Line2D([0], [0], color=BASE_GROUP_COLORS[group_name], lw=4, label=legend_label))
    dimension_legend = fig.legend(
        handles=dimension_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        ncol=3,
        title="Dimension",
    )
    plt.setp(dimension_legend.get_title(), fontsize=12, fontweight="bold")
    for text in dimension_legend.get_texts():
        text.set_fontsize(10)

    fig.subplots_adjust(left=0.13, right=0.78, top=0.86, bottom=0.10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_curve_histograms(output_path: str, matrix: np.ndarray, meta: Dict[str, object], style_meta: Dict[str, object], mode: str = "paper") -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x_spacing = 0.56
    x = np.arange(matrix.shape[1], dtype=np.float32) * x_spacing
    layout = build_ridge_layout(matrix, list(meta["row_labels"]), intra_gap=0.46, inter_gap=0.72)
    baselines = layout["baselines"]
    group_centers = layout["group_centers"]

    bar_width = 0.30
    for row_idx, label in enumerate(meta["row_labels"]):
        y = matrix[row_idx]
        base = baselines[row_idx]
        curve_color = style_meta["curve_colors"][row_idx]
        ax.bar(
            x,
            y,
            width=bar_width,
            bottom=base,
            color=curve_color,
            edgecolor=curve_color,
            linewidth=0.25,
            alpha=0.82 if mode == "paper" else 0.9,
            align="center",
        )
        ax.hlines(base, x[0] - 0.18, x[-1] + 0.18, colors=(0.2, 0.2, 0.2, 0.10), linewidth=0.8)

    ax.set_xlim(-0.20, x[-1] + 0.20)
    ax.set_xlabel("Feature Position")
    ax.set_ylabel("")
    ax.set_yticks([])

    if mode == "paper":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", alpha=0.08, linewidth=0.7)
    else:
        ax.grid(alpha=0.18, linewidth=0.8)

    for group_name, center_y in group_centers.items():
        ax.text(
            -0.92,
            center_y + 0.15,
            style_meta["legend_labels"][group_name],
            color=BASE_GROUP_COLORS[group_name],
            fontsize=9 if mode == "paper" else 8,
            fontweight="bold",
            ha="right",
            va="center",
            clip_on=False,
        )

    feature_handles = []
    for curve_label, curve_color in zip(style_meta["curve_labels"], style_meta["curve_colors"]):
        feature_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=curve_color, edgecolor=curve_color, label=curve_label))
    feature_legend = ax.legend(
        handles=feature_handles,
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        frameon=False,
        title="Feature",
        ncol=1,
    )
    plt.setp(feature_legend.get_title(), fontsize=11, fontweight="bold")
    for text in feature_legend.get_texts():
        text.set_fontsize(9)

    dimension_handles = []
    for group_name, legend_label in style_meta["legend_labels"].items():
        dimension_handles.append(plt.Line2D([0], [0], color=BASE_GROUP_COLORS[group_name], lw=5, label=legend_label))
    dimension_legend = fig.legend(
        handles=dimension_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        ncol=3,
        title="Dimension",
    )
    plt.setp(dimension_legend.get_title(), fontsize=12, fontweight="bold")
    for text in dimension_legend.get_texts():
        text.set_fontsize(10)

    fig.subplots_adjust(left=0.11, right=0.80, top=0.86, bottom=0.12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pick_image_rel(requested_image_rel: str, indexed_bundles: List[Dict[str, Dict[str, object]]]) -> str:
    if requested_image_rel:
        return requested_image_rel
    common = set(indexed_bundles[0].keys())
    for bundle in indexed_bundles[1:]:
        common &= set(bundle.keys())
    if not common:
        raise ValueError("No shared image_rel found across processed bundles.")
    return sorted(common)[0]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a 2D overlapped feature-curve figure for one processed sample.")
    parser.add_argument("--image-rel", default=None)
    parser.add_argument("--quality-path", default="./data/data_feature/quality/quality_processed_features.npy")
    parser.add_argument("--difficulty-path", default="./data/data_feature/difficulty/difficulty_processed_features.npy")
    parser.add_argument("--coverage-path", default="./data/data_feature/coverage/coverage_processed_features.npy")
    parser.add_argument("--output-dir", default="./figures/feature_curve_overlay")
    parser.add_argument("--mode", choices=["paper", "analysis"], default="paper")
    parser.add_argument("--style", choices=["ridge", "line", "hist"], default="ridge")
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
    style_meta = build_curve_style_metadata(list(meta["row_labels"]))

    os.makedirs(args.output_dir, exist_ok=True)
    sample_stem = os.path.splitext(os.path.basename(image_rel))[0]
    png_path = os.path.join(args.output_dir, f"{sample_stem}_feature_curve_overlay_{args.style}.png")
    if args.style == "line":
        render_curve_overlay(png_path, matrix, meta, style_meta, mode=args.mode)
    elif args.style == "hist":
        render_curve_histograms(png_path, matrix, meta, style_meta, mode=args.mode)
    else:
        render_curve_ridges(png_path, matrix, meta, style_meta, mode=args.mode)

    if args.save_matrix_json:
        json_path = os.path.join(args.output_dir, f"{sample_stem}_feature_curve_overlay_{args.style}.json")
        export_curve_overlay_json(json_path, matrix, meta, style_meta)
        print(f"[feature-curve-overlay] wrote {json_path}")

    print(f"[feature-curve-overlay] wrote {png_path}")


if __name__ == "__main__":
    main()
