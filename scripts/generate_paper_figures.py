from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_OUTPUT = REPO_ROOT / "assets" / "scenario_mining_results_overview.png"
PERCEPTION_ASSET_OUTPUT = REPO_ROOT / "assets" / "perception_slice_results_overview.png"


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _profile_order(rows: List[Dict[str, object]], preferred: List[str]) -> List[Dict[str, object]]:
    order_map = {name: idx for idx, name in enumerate(preferred)}
    return sorted(rows, key=lambda item: order_map.get(str(item.get("name") or ""), 999))


def _scenario_profile_rows() -> List[Dict[str, object]]:
    data = _load_json(REPO_ROOT / "outputs" / "trainval_scenario_profile_comparison_v1" / "benchmark_profile_comparison.json")
    rows = _profile_order(
        list(data.get("profiles") or []),
        ["rule_only", "llm_planner", "hybrid_agent"],
    )
    result: List[Dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "name": str(row["name"]),
                "label": str(row["label"]),
                "scene_at_1": float(row["scene_objective_at_1_rate"]),
                "reference_at_1": float(row["reference_objective_at_1_rate"]),
                "group_success_at_1": _ratio(
                    int(row["scenario_group_scene_success_at_1_count"]),
                    int(row["scenario_group_count"]),
                ),
                "event_iou": float(row["mean_event_iou"]),
            }
        )
    return result


def _ablation_rows() -> List[Dict[str, object]]:
    data = _load_json(REPO_ROOT / "outputs" / "trainval_hybrid_ablation_v1" / "benchmark_profile_comparison.json")
    rows = _profile_order(
        list(data.get("profiles") or []),
        ["full_system", "no_rerank", "no_map_context", "no_event_localization"],
    )
    result: List[Dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "name": str(row["name"]),
                "label": str(row["label"]),
                "scene_at_1": float(row["scene_objective_at_1_rate"]),
                "reference_at_1": float(row["reference_objective_at_1_rate"]),
                "group_success_at_1": _ratio(
                    int(row["scenario_group_scene_success_at_1_count"]),
                    int(row["scenario_group_count"]),
                ),
                "event_iou": float(row["mean_event_iou"]),
            }
        )
    return result


def _ablation_behavior_matrix() -> Dict[str, object]:
    rows = list(_load_json(REPO_ROOT / "outputs" / "trainval_hybrid_ablation_v1" / "behavior_error_analysis.json"))
    behavior_order = ["stopped_lead", "crossing", "oncoming", "none", "cut_in"]
    profile_order = ["full_system", "no_rerank", "no_map_context", "no_event_localization"]
    profile_labels = {
        "full_system": "Full",
        "no_rerank": "No-Rerank",
        "no_map_context": "No-Map",
        "no_event_localization": "No-Event",
    }
    behavior_map = {str(row["behavior"]): row for row in rows}
    matrix = np.zeros((len(behavior_order), len(profile_order)), dtype=float)
    for i, behavior in enumerate(behavior_order):
        row = dict(behavior_map[behavior])
        summary = dict(row["profile_summary"])
        for j, profile_name in enumerate(profile_order):
            matrix[i, j] = float(summary[profile_name]["scene_objective_at_1_rate"])
    return {
        "behaviors": behavior_order,
        "profiles": profile_order,
        "profile_labels": [profile_labels[name] for name in profile_order],
        "matrix": matrix,
    }


def _perception_slice_rows() -> List[Dict[str, object]]:
    data = _load_json(REPO_ROOT / "outputs" / "trainval_perception_proxy_study_v1" / "perception_comparison.json")
    rows = _profile_order(
        list(data.get("profiles") or []),
        ["oracle_tracking", "crossing_sparse_track", "delayed_track"],
    )
    result: List[Dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "name": str(row["name"]),
                "label": str(row["label"]),
                "full_track_rate": float(row["full_track_rate"]),
                "mean_event_recall": float(row["mean_event_recall"]),
                "mean_contiguous_coverage": float(row["mean_contiguous_coverage"]),
                "mean_center_error_m": float(row["mean_center_error_m"]),
            }
        )
    return result


def _perception_benchmark_composition() -> Dict[str, object]:
    payload = _load_json(REPO_ROOT / "benchmarks" / "trainval_perception_slices_v1.json")
    behavior_order = ["stopped_lead", "crossing", "oncoming", "proximity", "cut_in"]
    counts = {name: 0 for name in behavior_order}
    frame_counts: List[int] = []
    for case in list(payload.get("cases") or []):
        behavior = str(case.get("primary_behavior") or "proximity")
        counts.setdefault(behavior, 0)
        counts[behavior] += 1
        frame_counts.append(int(case.get("frame_count") or 0))
    return {
        "behaviors": behavior_order,
        "counts": [counts.get(name, 0) for name in behavior_order],
        "mean_frame_count": float(np.mean(frame_counts)) if frame_counts else 0.0,
    }


def _perception_behavior_matrix() -> Dict[str, object]:
    data = _load_json(REPO_ROOT / "outputs" / "trainval_perception_proxy_study_v1" / "perception_comparison.json")
    rows = list(data.get("behavior_matrix") or [])
    behavior_order = ["stopped_lead", "crossing", "oncoming", "proximity", "cut_in"]
    profile_order = ["oracle_tracking", "crossing_sparse_track", "delayed_track"]
    profile_labels = {
        "oracle_tracking": "Oracle",
        "crossing_sparse_track": "Sparse",
        "delayed_track": "Delayed",
    }
    behavior_map = {str(row["behavior"]): row for row in rows}
    matrix = np.zeros((len(behavior_order), len(profile_order)), dtype=float)

    profiles = list(data.get("profiles") or [])
    profile_index = {str(row["name"]): idx for idx, row in enumerate(profiles)}
    for i, behavior in enumerate(behavior_order):
        row = dict(behavior_map.get(behavior) or {})
        cells = list(row.get("cells") or [])
        for j, profile_name in enumerate(profile_order):
            cell_idx = profile_index.get(profile_name)
            if cell_idx is None or cell_idx >= len(cells):
                continue
            matrix[i, j] = float(cells[cell_idx].get("mean_event_recall") or 0.0)
    return {
        "behaviors": behavior_order,
        "profile_labels": [profile_labels[name] for name in profile_order],
        "matrix": matrix,
    }


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.facecolor": "#fffdfa",
            "figure.facecolor": "#f7f3ec",
            "savefig.facecolor": "#f7f3ec",
            "axes.edgecolor": "#cfc3b2",
            "grid.color": "#ddd3c4",
            "grid.alpha": 0.6,
            "axes.grid": True,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_metric_panel(ax: plt.Axes, rows: List[Dict[str, object]], title: str) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.22
    colors = ["#264653", "#2a9d8f", "#c46a00"]

    scene = np.array([float(row["scene_at_1"]) * 100.0 for row in rows])
    reference = np.array([float(row["reference_at_1"]) * 100.0 for row in rows])
    group_success = np.array([float(row["group_success_at_1"]) * 100.0 for row in rows])
    event_iou = np.array([float(row["event_iou"]) for row in rows])

    ax.bar(x - width, scene, width=width, color=colors[0], label="Scene@1")
    ax.bar(x, reference, width=width, color=colors[1], label="Reference@1")
    ax.bar(x + width, group_success, width=width, color=colors[2], label="Group Success@1")
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)

    line_ax = ax.twinx()
    line_ax.plot(x, event_iou, color="#7a1f1f", marker="o", linewidth=2.0, label="Event IoU")
    line_ax.set_ylim(0.0, 1.05)
    line_ax.set_ylabel("Event IoU")
    line_ax.grid(False)

    handles, labels_left = ax.get_legend_handles_labels()
    handles2, labels_right = line_ax.get_legend_handles_labels()
    ax.legend(handles + handles2, labels_left + labels_right, loc="upper right", frameon=False)

    for idx, value in enumerate(scene):
        ax.text(idx - width, value + 1.5, "{0:.1f}".format(value), ha="center", va="bottom", fontsize=9, color=colors[0])
    for idx, value in enumerate(reference):
        ax.text(idx, value + 1.5, "{0:.1f}".format(value), ha="center", va="bottom", fontsize=9, color=colors[1])
    for idx, value in enumerate(group_success):
        ax.text(idx + width, value + 1.5, "{0:.1f}".format(value), ha="center", va="bottom", fontsize=9, color=colors[2])


def _plot_heatmap(ax: plt.Axes, matrix_payload: Dict[str, object]) -> None:
    matrix = np.asarray(matrix_payload["matrix"], dtype=float)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("Behavior-Wise Scene@1 Under Ablation")
    ax.set_xticks(np.arange(len(matrix_payload["profile_labels"])))
    ax.set_xticklabels(list(matrix_payload["profile_labels"]))
    ax.set_yticks(np.arange(len(matrix_payload["behaviors"])))
    ax.set_yticklabels(list(matrix_payload["behaviors"]))
    ax.set_xlabel("Ablation Variant")
    ax.set_ylabel("Behavior Family")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                "{0:.0f}".format(value * 100.0),
                ha="center",
                va="center",
                color="white" if value < 0.55 else "#143642",
                fontsize=10,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Scene@1")


def _plot_perception_composition(ax: plt.Axes, payload: Dict[str, object]) -> None:
    labels = [name.replace("_", "\n") for name in payload["behaviors"]]
    values = np.asarray(payload["counts"], dtype=float)
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#c46a00", "#7a1f1f"]
    ax.bar(np.arange(len(labels)), values, color=colors[: len(labels)], width=0.64)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Case Count")
    ax.set_title("Perception Slice Composition")
    ax.set_ylim(0.0, max(float(values.max()) + 1.2, 3.5))
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.08, "{0:.0f}".format(value), ha="center", va="bottom", fontsize=10, color="#1f1f1f")
    ax.text(
        0.02,
        0.96,
        "Mean window length: {0:.1f} frames".format(float(payload["mean_frame_count"])),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#665d54",
    )


def _plot_perception_profiles(ax: plt.Axes, rows: List[Dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.22
    colors = ["#264653", "#2a9d8f", "#c46a00"]
    full_track = np.array([float(row["full_track_rate"]) * 100.0 for row in rows])
    event_recall = np.array([float(row["mean_event_recall"]) * 100.0 for row in rows])
    contiguous = np.array([float(row["mean_contiguous_coverage"]) * 100.0 for row in rows])
    center_error = np.array([float(row["mean_center_error_m"]) for row in rows])

    ax.bar(x - width, full_track, width=width, color=colors[0], label="Full Track")
    ax.bar(x, event_recall, width=width, color=colors[1], label="Event Recall")
    ax.bar(x + width, contiguous, width=width, color=colors[2], label="Contiguous Coverage")
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Proxy Perception Profile Comparison")

    line_ax = ax.twinx()
    line_ax.plot(x, center_error, color="#7a1f1f", marker="o", linewidth=2.0, label="Center Error")
    line_ax.set_ylim(0.0, max(float(center_error.max()) + 0.15, 0.6))
    line_ax.set_ylabel("Center Error (m)")
    line_ax.grid(False)

    handles, labels_left = ax.get_legend_handles_labels()
    handles2, labels_right = line_ax.get_legend_handles_labels()
    ax.legend(handles + handles2, labels_left + labels_right, loc="upper right", frameon=False)

    for idx, value in enumerate(event_recall):
        ax.text(idx, value + 1.5, "{0:.1f}".format(value), ha="center", va="bottom", fontsize=9, color=colors[1])


def _plot_perception_heatmap(ax: plt.Axes, matrix_payload: Dict[str, object]) -> None:
    matrix = np.asarray(matrix_payload["matrix"], dtype=float)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("Behavior-Wise Event Recall")
    ax.set_xticks(np.arange(len(matrix_payload["profile_labels"])))
    ax.set_xticklabels(list(matrix_payload["profile_labels"]))
    ax.set_yticks(np.arange(len(matrix_payload["behaviors"])))
    ax.set_yticklabels(list(matrix_payload["behaviors"]))
    ax.set_xlabel("Proxy Profile")
    ax.set_ylabel("Behavior Family")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                "{0:.0f}".format(value * 100.0),
                ha="center",
                va="center",
                color="white" if value < 0.55 else "#143642",
                fontsize=10,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Event Recall")


def main() -> None:
    _apply_style()
    scenario_rows = _scenario_profile_rows()
    ablation_rows = _ablation_rows()
    matrix_payload = _ablation_behavior_matrix()

    fig = plt.figure(figsize=(14.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    _plot_metric_panel(ax1, scenario_rows, "Scenario Mining Comparison")
    _plot_metric_panel(ax2, ablation_rows, "Hybrid Ablation")
    _plot_heatmap(ax3, matrix_payload)

    fig.suptitle(
        "nuScenes Scene Mining: Profile Comparison and Hybrid Ablation",
        fontsize=16,
        fontweight="bold",
        color="#1f1f1f",
    )
    fig.savefig(ASSET_OUTPUT, dpi=220, bbox_inches="tight")

    perception_rows = _perception_slice_rows()
    perception_composition = _perception_benchmark_composition()
    perception_matrix = _perception_behavior_matrix()
    fig2 = plt.figure(figsize=(16.0, 5.6), constrained_layout=True)
    grid2 = fig2.add_gridspec(1, 3, width_ratios=[0.9, 1.35, 1.15])
    pax1 = fig2.add_subplot(grid2[0, 0])
    pax2 = fig2.add_subplot(grid2[0, 1])
    pax3 = fig2.add_subplot(grid2[0, 2])

    _plot_perception_composition(pax1, perception_composition)
    _plot_perception_profiles(pax2, perception_rows)
    _plot_perception_heatmap(pax3, perception_matrix)

    fig2.suptitle(
        "nuScenes Scene Mining: Scenario-Conditioned Perception Slice Evaluation",
        fontsize=16,
        fontweight="bold",
        color="#1f1f1f",
    )
    fig2.savefig(PERCEPTION_ASSET_OUTPUT, dpi=220, bbox_inches="tight")
    print(ASSET_OUTPUT)
    print(PERCEPTION_ASSET_OUTPUT)


if __name__ == "__main__":
    main()
