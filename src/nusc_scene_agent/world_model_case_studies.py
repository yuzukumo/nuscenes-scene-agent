from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BACKGROUND = "#f5f1ea"
GRID_COLOR = "#d8d2ca"
HISTORY_COLOR = "#202124"
GT_COLOR = "#d95f02"
GT_OCCUPANCY_COLOR = "#f7c59f"
CONTEXT_COLOR = "#c9c3bb"
PROFILE_COLORS = ["#264653", "#2a9d8f", "#c46a00", "#7a1f1f", "#6b8e23"]


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _unique_strings(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _profile_label(name: str) -> str:
    return str(name).replace("_", "-").title()


def _prediction_index(path: Path) -> Dict[str, Dict[str, object]]:
    payload = _load_json(path)
    return {
        str(item.get("benchmark_group") or ""): dict(item)
        for item in list(payload.get("predictions") or [])
        if str(item.get("benchmark_group") or "")
    }


def _metrics_index(path: Path) -> Dict[str, Dict[str, object]]:
    payload = _load_json(path)
    return {
        str(item.get("benchmark_group") or ""): dict(item)
        for item in list(payload.get("case_metrics") or [])
        if str(item.get("benchmark_group") or "")
    }


def _case_study_rows(
    benchmark_path: Path,
    evaluation_dirs: Sequence[Path],
) -> List[Dict[str, object]]:
    benchmark = _load_json(benchmark_path)
    evaluation_payloads = []
    for eval_dir in evaluation_dirs:
        eval_dir = Path(eval_dir).resolve()
        metrics_path = eval_dir / "world_model_metrics.json"
        if not metrics_path.exists():
            continue
        metrics_payload = _load_json(metrics_path)
        profile_name = str(metrics_payload.get("profile_name") or eval_dir.name)
        predictions_path = eval_dir / "adapted_predictions.json"
        if not predictions_path.exists():
            fallback_predictions_path = eval_dir.parent / "predictions" / "{0}.json".format(profile_name)
            if fallback_predictions_path.exists():
                predictions_path = fallback_predictions_path
        if not predictions_path.exists():
            fallback_predictions_path = eval_dir.parent / "{0}.json".format(profile_name)
            if fallback_predictions_path.exists():
                predictions_path = fallback_predictions_path
        if not predictions_path.exists():
            continue
        evaluation_payloads.append(
            {
                "profile_name": profile_name,
                "metrics": _metrics_index(metrics_path),
                "predictions": _prediction_index(predictions_path),
            }
        )

    rows = []
    for case in list(benchmark.get("cases") or []):
        benchmark_group = str(case.get("benchmark_group") or "")
        profiles = []
        risk_scores = []
        ade_values = []
        for payload in evaluation_payloads:
            metric = dict(payload["metrics"].get(benchmark_group) or {})
            prediction = dict(payload["predictions"].get(benchmark_group) or {})
            if not metric:
                continue
            risk_score = _safe_float(metric.get("risk_fidelity_score"), 0.0)
            ade_m = _safe_float(metric.get("ade_m"), float("nan"))
            profiles.append(
                {
                    "profile_name": str(payload["profile_name"]),
                    "label": _profile_label(str(payload["profile_name"])),
                    "metric": metric,
                    "prediction": prediction,
                }
            )
            risk_scores.append(risk_score)
            if math.isfinite(ade_m):
                ade_values.append(ade_m)
        if not profiles:
            continue
        rows.append(
            {
                "case": dict(case),
                "profiles": profiles,
                "risk_gap": max(risk_scores) - min(risk_scores) if risk_scores else 0.0,
                "mean_ade_m": mean(ade_values) if ade_values else 0.0,
            }
        )
    return rows


def _select_case_studies(case_rows: Sequence[Dict[str, object]], max_cases: int) -> List[Dict[str, object]]:
    track_seen = set()
    selected = []
    ordered = sorted(
        case_rows,
        key=lambda row: (
            float(row["risk_gap"]),
            float(row["mean_ade_m"]),
            str(row["case"].get("primary_behavior") or ""),
        ),
        reverse=True,
    )

    for row in ordered:
        tracks = _unique_strings(list(row["case"].get("challenge_tracks") or []))
        primary_track = tracks[0] if tracks else "challenge/generic_risk_slice"
        if primary_track in track_seen and len(selected) < max_cases // 2:
            continue
        track_seen.add(primary_track)
        selected.append(row)
        if len(selected) >= max_cases:
            break

    if len(selected) < min(max_cases, len(ordered)):
        for row in ordered:
            if row in selected:
                continue
            selected.append(row)
            if len(selected) >= max_cases:
                break
    return selected


def _cell_center(cell: Sequence[object], grid_spec: Dict[str, object]) -> np.ndarray:
    resolution = _safe_float(grid_spec.get("resolution_m"), 1.0)
    x_min = _safe_float(grid_spec.get("x_min_m"), -24.0)
    y_min = _safe_float(grid_spec.get("y_min_m"), -24.0)
    x_idx = int(cell[0])
    y_idx = int(cell[1])
    return np.asarray(
        [
            x_min + (x_idx + 0.5) * resolution,
            y_min + (y_idx + 0.5) * resolution,
        ],
        dtype=float,
    )


def _draw_case_panel(ax: plt.Axes, row: Dict[str, object], grid_spec: Dict[str, object], color_map: Dict[str, str]) -> None:
    case = dict(row["case"])
    history_frames = list(case.get("history_frames") or [])
    future_frames = list(case.get("future_frames") or [])
    future_occupancy = list(case.get("future_occupancy") or [])

    ax.set_facecolor(BACKGROUND)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.axhline(0.0, color="#99948c", linewidth=0.8, alpha=0.8)
    ax.axvline(0.0, color="#99948c", linewidth=0.8, alpha=0.8)

    for frame in future_occupancy:
        for cell in list(frame.get("context_cells") or []):
            center = _cell_center(cell, grid_spec)
            ax.scatter(center[0], center[1], marker="s", s=16, color=CONTEXT_COLOR, alpha=0.22, linewidths=0)
        for cell in list(frame.get("primary_actor_cells") or []):
            center = _cell_center(cell, grid_spec)
            ax.scatter(center[0], center[1], marker="s", s=20, color=GT_OCCUPANCY_COLOR, alpha=0.34, linewidths=0)

    if history_frames:
        history_xy = np.asarray([[float(frame["x_ego"]), float(frame["y_ego"])] for frame in history_frames], dtype=float)
        ax.plot(history_xy[:, 0], history_xy[:, 1], color=HISTORY_COLOR, linewidth=2.0, label="history")
        ax.scatter(history_xy[-1, 0], history_xy[-1, 1], color=HISTORY_COLOR, s=32, zorder=6)

    if future_frames:
        gt_xy = np.asarray([[float(frame["x_ego"]), float(frame["y_ego"])] for frame in future_frames], dtype=float)
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=GT_COLOR, linewidth=2.4, label="ground truth")
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1], color=GT_COLOR, s=20, zorder=6)

    for profile in list(row["profiles"]):
        prediction = dict(profile["prediction"])
        trajectory = list(prediction.get("future_trajectory") or [])
        if not trajectory:
            continue
        pred_xy = np.asarray([[float(frame["x_ego"]), float(frame["y_ego"])] for frame in trajectory], dtype=float)
        color = color_map[profile["profile_name"]]
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color=color, linewidth=2.0, linestyle="--", label=profile["label"])
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], color=color, s=18, zorder=6)

    tracks = ", ".join(_unique_strings(list(case.get("challenge_tracks") or [])))
    ax.set_title("{0} | {1}".format(case.get("scene_name") or "", case.get("primary_behavior") or ""))
    ax.set_xlim(-25.0, 25.0)
    ax.set_ylim(-25.0, 25.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward / x_ego (m)")
    ax.set_ylabel("Left / y_ego (m)")

    metric_lines = [
        "Tracks: {0}".format(tracks or "none"),
    ]
    for profile in list(row["profiles"]):
        metric = dict(profile["metric"])
        metric_lines.append(
            "{0}: rf={1:.3f}, ade={2:.3f}".format(
                profile["label"],
                _safe_float(metric.get("risk_fidelity_score"), 0.0),
                _safe_float(metric.get("ade_m"), 0.0),
            )
        )
    ax.text(
        0.02,
        0.02,
        "\n".join(metric_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        color="#2f2b27",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffdfa", "edgecolor": "#ddd4c7", "alpha": 0.92},
    )


def render_world_model_case_studies(
    benchmark_path: Path,
    evaluation_dirs: Sequence[Path],
    output_dir: Path,
    max_cases: int = 4,
) -> Dict[str, object]:
    rows = _case_study_rows(benchmark_path, evaluation_dirs)
    selected = _select_case_studies(rows, max_cases=max_cases)
    benchmark = _load_json(benchmark_path)
    grid_spec = dict((benchmark.get("metadata") or {}).get("grid_spec") or {})

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_names = _unique_strings(
        [str(profile["profile_name"]) for row in selected for profile in list(row.get("profiles") or [])]
    )
    color_map = {
        name: PROFILE_COLORS[idx % len(PROFILE_COLORS)]
        for idx, name in enumerate(profile_names)
    }

    cols = 2 if len(selected) > 1 else 1
    rows_n = int(math.ceil(len(selected) / cols)) if selected else 1
    fig = plt.figure(figsize=(8.8 * cols, 6.8 * rows_n), constrained_layout=True)
    grid = fig.add_gridspec(rows_n, cols)
    axes = []
    for idx in range(rows_n * cols):
        axes.append(fig.add_subplot(grid[idx // cols, idx % cols]))

    for ax, row in zip(axes, selected):
        _draw_case_panel(ax, row, grid_spec=grid_spec, color_map=color_map)
    for ax in axes[len(selected):]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color=HISTORY_COLOR, linewidth=2.0, label="History"),
        plt.Line2D([0], [0], color=GT_COLOR, linewidth=2.4, label="Ground Truth"),
    ]
    for name in profile_names:
        legend_handles.append(
            plt.Line2D([0], [0], color=color_map[name], linewidth=2.0, linestyle="--", label=_profile_label(name))
        )
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle("Scenario-Conditioned World-Model Case Studies", fontsize=16, fontweight="bold", color="#1f1f1f")

    figure_path = output_dir / "world_model_case_studies.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for row in selected:
        case = dict(row["case"])
        summary_rows.append(
            {
                "benchmark_group": str(case.get("benchmark_group") or ""),
                "scene_name": str(case.get("scene_name") or ""),
                "primary_behavior": str(case.get("primary_behavior") or ""),
                "challenge_tracks": list(case.get("challenge_tracks") or []),
                "profiles": [
                    {
                        "profile_name": str(profile["profile_name"]),
                        "risk_fidelity_score": _safe_float(profile["metric"].get("risk_fidelity_score"), 0.0),
                        "ade_m": _safe_float(profile["metric"].get("ade_m"), 0.0),
                        "occupancy_iou": _safe_float(profile["metric"].get("occupancy_iou"), 0.0),
                    }
                    for profile in list(row["profiles"])
                ],
            }
        )

    metadata = {
        "benchmark_path": str(benchmark_path),
        "evaluation_dirs": [str(Path(path).resolve()) for path in evaluation_dirs],
        "case_count": len(summary_rows),
        "figure_path": str(figure_path),
        "cases": summary_rows,
    }
    (output_dir / "world_model_case_studies.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Scenario-Conditioned World-Model Case Studies",
        "",
        "- Cases: {0}".format(len(summary_rows)),
        "- Figure: `world_model_case_studies.png`",
        "",
        "| Scene | Behavior | Tracks | Profile Summary |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        profile_summary = "; ".join(
            "{0}: rf={1:.3f}, ade={2:.3f}".format(
                _profile_label(profile["profile_name"]),
                float(profile["risk_fidelity_score"]),
                float(profile["ade_m"]),
            )
            for profile in list(row["profiles"])
        )
        lines.append(
            "| {0} | {1} | {2} | {3} |".format(
                row["scene_name"],
                row["primary_behavior"],
                ", ".join(row["challenge_tracks"]),
                profile_summary,
            )
        )
    (output_dir / "world_model_case_studies.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "case_count": len(summary_rows),
        "figure_path": str(figure_path),
    }
