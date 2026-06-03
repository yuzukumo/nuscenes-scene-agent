from __future__ import annotations

import csv
import html
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from nusc_scene_agent.perception_benchmark import CATEGORY_ALIASES, RISK_FACET_FIELDS


DEFAULT_BEV_OCCUPANCY_BENCHMARK = Path("benchmarks/trainval_bev_occupancy_slices_v1.json")

DEFAULT_BEV_GRID_SPEC = {
    "x_min_m": -40.0,
    "x_max_m": 80.0,
    "y_min_m": -40.0,
    "y_max_m": 40.0,
    "resolution_m": 1.0,
    "dilation_radius_cells": 1.0,
}

BEV_OCCUPANCY_PROXY_PROFILES = [
    "oracle_occupancy",
    "risk_actor_only",
    "context_drop_occupancy",
]


def generate_bev_occupancy_benchmark_from_perception_benchmark(
    perception_benchmark_path: Path,
    db_path: Path,
    output_path: Path = DEFAULT_BEV_OCCUPANCY_BENCHMARK,
    *,
    grid_spec: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    perception_payload = _read_json(Path(perception_benchmark_path))
    grid = _normalize_grid_spec(grid_spec)
    cases = [dict(case) for case in list(perception_payload.get("cases") or [])]
    sample_tokens = _unique_strings(
        str(frame.get("sample_token") or "")
        for case in cases
        for frame in list(case.get("frames") or [])
    )

    conn = sqlite3.connect(str(Path(db_path).resolve()))
    conn.row_factory = sqlite3.Row
    try:
        context_index = _load_agent_context(conn, sample_tokens)
    finally:
        conn.close()

    occupancy_cases = []
    for case in cases:
        instance_token = str(case.get("instance_token") or "")
        frames = []
        for frame in list(case.get("frames") or []):
            sample_token = str(frame.get("sample_token") or "")
            sample_rows = list(context_index.get(sample_token) or [])
            primary_cells = _rasterize_points([(_safe_float(frame.get("x_ego")), _safe_float(frame.get("y_ego")))], grid)
            context_cells = _rasterize_points(
                [
                    (_safe_float(row.get("x_ego")), _safe_float(row.get("y_ego")))
                    for row in sample_rows
                    if str(row.get("instance_token") or "") != instance_token
                ],
                grid,
            )
            frames.append(
                {
                    "sample_token": sample_token,
                    "sample_idx": _safe_int(frame.get("sample_idx")),
                    "timestamp_us": _safe_int(frame.get("timestamp_us")),
                    "primary_actor_cells": primary_cells,
                    "context_cells": context_cells,
                    "occupied_cells": _union_cells(primary_cells, context_cells),
                    "context_actor_count": sum(
                        1 for row in sample_rows if str(row.get("instance_token") or "") != instance_token
                    ),
                }
            )
        occupancy_cases.append(
            {
                "benchmark_group": str(case.get("benchmark_group") or ""),
                "reference_case_key": str(case.get("reference_case_key") or ""),
                "reference_scene_name": str(case.get("reference_scene_name") or ""),
                "scene_name": str(case.get("scene_name") or ""),
                "scene_token": str(case.get("scene_token") or ""),
                "instance_token": instance_token,
                "category_group": _normalize_category(str(case.get("category_group") or "")),
                "location": str(case.get("location") or ""),
                "primary_behavior": str(case.get("primary_behavior") or ""),
                "behaviors": list(case.get("behaviors") or []),
                "tags": list(case.get("tags") or []),
                "anchor_sample_token": str(case.get("anchor_sample_token") or ""),
                "anchor_sample_idx": _safe_int(case.get("anchor_sample_idx")),
                "event_start_sample_idx": _safe_int(case.get("event_start_sample_idx")),
                "event_end_sample_idx": _safe_int(case.get("event_end_sample_idx")),
                "event_peak_sample_idx": _safe_int(case.get("event_peak_sample_idx")),
                "frame_count": len(frames),
                "frames": frames,
                "risk_facets": dict(case.get("risk_facets") or {}),
                "map_support": dict(case.get("map_support") or {}),
                "min_distance_m": case.get("min_distance_m"),
                "min_ttc_s": case.get("min_ttc_s"),
            }
        )

    output = {
        "metadata": {
            "schema": "risk_bev_occupancy_benchmark_v1",
            "generator": "bev_occupancy_benchmark_generator_v1",
            "source_perception_benchmark": str(perception_benchmark_path),
            "db_path": str(db_path),
            "label_type": "sparse_center_cell_occupancy",
            "case_count": len(occupancy_cases),
            "grid_spec": grid,
            "case_distribution": _case_distribution(occupancy_cases),
        },
        "cases": occupancy_cases,
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_path": str(output_path), "case_count": len(occupancy_cases), "schema": output["metadata"]["schema"]}


def generate_proxy_bev_occupancy_predictions(
    benchmark_path: Path,
    output_path: Path,
    profile_name: str,
) -> Dict[str, Any]:
    if profile_name not in BEV_OCCUPANCY_PROXY_PROFILES:
        raise ValueError("Unknown BEV occupancy profile: {0}".format(profile_name))

    benchmark = _read_json(Path(benchmark_path))
    predictions = []
    for case in list(benchmark.get("cases") or []):
        frames = []
        for frame in list(case.get("frames") or []):
            primary_cells = list(frame.get("primary_actor_cells") or [])
            context_cells = list(frame.get("context_cells") or [])
            if profile_name == "oracle_occupancy":
                predicted_primary = primary_cells
                predicted_context = context_cells
            elif profile_name == "risk_actor_only":
                predicted_primary = primary_cells
                predicted_context = []
            else:
                predicted_primary = primary_cells
                predicted_context = [
                    cell
                    for idx, cell in enumerate(context_cells)
                    if (idx + _safe_int(frame.get("sample_idx"))) % 2 == 0
                ]
            frames.append(
                {
                    "sample_token": str(frame.get("sample_token") or ""),
                    "sample_idx": _safe_int(frame.get("sample_idx")),
                    "primary_actor_cells": predicted_primary,
                    "context_cells": predicted_context,
                    "occupied_cells": _union_cells(predicted_primary, predicted_context),
                }
            )
        predictions.append(
            {
                "benchmark_group": str(case.get("benchmark_group") or ""),
                "frames": frames,
            }
        )

    output = {
        "metadata": {
            "generator": "proxy_bev_occupancy_predictions_v1",
            "profile_name": profile_name,
            "source_benchmark": str(benchmark_path),
        },
        "predictions": predictions,
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_path": str(output_path), "profile_name": profile_name, "prediction_count": len(predictions)}


def adapt_perception_predictions_to_bev_occupancy(
    benchmark_path: Path,
    perception_predictions_path: Path,
    output_path: Path,
    *,
    profile_name: str = "",
) -> Dict[str, Any]:
    benchmark = _read_json(Path(benchmark_path))
    predictions = _read_json(Path(perception_predictions_path))
    grid = dict((benchmark.get("metadata") or {}).get("grid_spec") or DEFAULT_BEV_GRID_SPEC)
    rows_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in list(predictions.get("predictions") or []):
        rows_by_sample[str(row.get("sample_token") or "")].append(dict(row))

    adapted = []
    for case in list(benchmark.get("cases") or []):
        frames = []
        for frame in list(case.get("frames") or []):
            sample_token = str(frame.get("sample_token") or "")
            cells = _rasterize_points(
                [
                    (_safe_float(row.get("x_ego")), _safe_float(row.get("y_ego")))
                    for row in rows_by_sample.get(sample_token, [])
                ],
                grid,
            )
            frames.append(
                {
                    "sample_token": sample_token,
                    "sample_idx": _safe_int(frame.get("sample_idx")),
                    "occupied_cells": cells,
                    "primary_actor_cells": [],
                    "context_cells": [],
                }
            )
        adapted.append({"benchmark_group": str(case.get("benchmark_group") or ""), "frames": frames})

    profile = profile_name or str((predictions.get("metadata") or {}).get("profile_name") or Path(perception_predictions_path).stem)
    output = {
        "metadata": {
            "generator": "perception_to_bev_occupancy_adapter_v1",
            "profile_name": profile,
            "source_benchmark": str(benchmark_path),
            "source_predictions": str(perception_predictions_path),
        },
        "predictions": adapted,
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_path": str(output_path), "profile_name": profile, "prediction_count": len(adapted)}


def evaluate_bev_occupancy_predictions(
    benchmark_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    profile_name: str = "",
) -> Dict[str, Any]:
    benchmark = _read_json(Path(benchmark_path))
    predictions = _read_json(Path(predictions_path))
    profile = profile_name or str((predictions.get("metadata") or {}).get("profile_name") or Path(predictions_path).stem)
    prediction_index = _build_prediction_index(list(predictions.get("predictions") or []))

    case_metrics = []
    for case in list(benchmark.get("cases") or []):
        row = _evaluate_case(case, prediction_index)
        support = dict(case.get("risk_facets") or {})
        row.update({name: str(support.get(name) or "unknown") for name in RISK_FACET_FIELDS})
        row["min_distance_m"] = case.get("min_distance_m")
        row["min_ttc_s"] = case.get("min_ttc_s")
        case_metrics.append(row)

    summary = _build_summary(
        profile_name=profile,
        benchmark_path=Path(benchmark_path),
        predictions_path=Path(predictions_path),
        case_metrics=case_metrics,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bev_occupancy_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "bev_occupancy_metrics_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    (output_dir / "bev_occupancy_metrics_summary.html").write_text(
        _markdown_to_basic_html(_render_markdown(summary)),
        encoding="utf-8",
    )
    _write_case_metrics_csv(summary["case_metrics"], output_dir / "bev_occupancy_case_metrics.csv")
    return summary


def compare_bev_occupancy_evaluations(evaluation_dirs: Sequence[Path], output_dir: Path) -> Dict[str, Any]:
    summaries = [_read_json(Path(path) / "bev_occupancy_metrics.json") for path in evaluation_dirs]
    comparison = build_bev_occupancy_comparison(summaries)
    write_bev_occupancy_comparison(comparison, output_dir)
    return {
        "output_dir": str(Path(output_dir).resolve()),
        "profile_count": int(comparison["overview"]["profile_count"]),
        "case_count": int(comparison["overview"]["case_count"]),
    }


def run_proxy_bev_occupancy_study(benchmark_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    evaluation_dirs = []
    for profile_name in BEV_OCCUPANCY_PROXY_PROFILES:
        prediction_path = output_dir / "predictions" / f"{profile_name}.json"
        generate_proxy_bev_occupancy_predictions(benchmark_path, prediction_path, profile_name)
        eval_dir = output_dir / profile_name
        summaries.append(evaluate_bev_occupancy_predictions(benchmark_path, prediction_path, eval_dir, profile_name=profile_name))
        evaluation_dirs.append(eval_dir)

    comparison = build_bev_occupancy_comparison(summaries)
    write_bev_occupancy_comparison(comparison, output_dir)
    return {
        "output_dir": str(output_dir),
        "benchmark_path": str(benchmark_path),
        "profile_count": len(summaries),
        "case_count": int(comparison["overview"]["case_count"]),
        "evaluation_dirs": [str(path) for path in evaluation_dirs],
    }


def build_bev_occupancy_comparison(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    profiles = []
    behavior_index: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for summary in summaries:
        overview = dict(summary.get("overview") or {})
        profile_name = str(summary.get("profile_name") or "profile")
        profiles.append(
            {
                "name": profile_name,
                "case_count": int(overview.get("case_count") or 0),
                "mean_occupancy_iou": float(overview.get("mean_occupancy_iou") or 0.0),
                "mean_primary_actor_recall": float(overview.get("mean_primary_actor_recall") or 0.0),
                "mean_context_recall": float(overview.get("mean_context_recall") or 0.0),
                "mean_anchor_occupancy_iou": float(overview.get("mean_anchor_occupancy_iou") or 0.0),
                "mean_risk_fidelity_score": float(overview.get("mean_risk_fidelity_score") or 0.0),
                "perfect_case_count": int(overview.get("perfect_case_count") or 0),
            }
        )
        for row in list(summary.get("behavior_breakdown") or []):
            behavior_index[str(row["primary_behavior"])][profile_name] = dict(row)

    profiles.sort(
        key=lambda row: (
            row["mean_risk_fidelity_score"],
            row["mean_primary_actor_recall"],
            row["mean_occupancy_iou"],
        ),
        reverse=True,
    )
    profile_order = [str(row["name"]) for row in profiles]
    behavior_matrix = []
    for behavior in sorted(behavior_index):
        behavior_matrix.append(
            {
                "primary_behavior": behavior,
                "cells": [
                    dict(
                        behavior_index[behavior].get(
                            name,
                            {
                                "case_count": 0,
                                "mean_occupancy_iou": 0.0,
                                "mean_primary_actor_recall": 0.0,
                                "mean_context_recall": 0.0,
                                "mean_risk_fidelity_score": 0.0,
                            },
                        )
                    )
                    for name in profile_order
                ],
            }
        )
    return {
        "metadata": {"schema": "bev_occupancy_comparison_v1"},
        "overview": {
            "profile_count": len(profiles),
            "case_count": int(profiles[0]["case_count"]) if profiles else 0,
        },
        "profiles": profiles,
        "behavior_matrix": behavior_matrix,
    }


def write_bev_occupancy_comparison(comparison: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bev_occupancy_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# BEV Occupancy Slice Comparison",
        "",
        f"- Profiles: {comparison['overview']['profile_count']}",
        f"- Cases: {comparison['overview']['case_count']}",
        "",
        "| Profile | Occupancy IoU | Primary Recall | Context Recall | Anchor IoU | Risk Fidelity | Perfect Cases |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(comparison.get("profiles") or []):
        lines.append(
            "| {0} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6}/{7} |".format(
                row["name"],
                float(row["mean_occupancy_iou"]),
                float(row["mean_primary_actor_recall"]),
                float(row["mean_context_recall"]),
                float(row["mean_anchor_occupancy_iou"]),
                float(row["mean_risk_fidelity_score"]),
                int(row["perfect_case_count"]),
                int(row["case_count"]),
            )
        )
    text = "\n".join(lines) + "\n"
    (output_dir / "bev_occupancy_comparison_summary.md").write_text(text, encoding="utf-8")
    (output_dir / "bev_occupancy_comparison_summary.html").write_text(_markdown_to_basic_html(text), encoding="utf-8")
    with (output_dir / "bev_occupancy_leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "name",
            "case_count",
            "mean_occupancy_iou",
            "mean_primary_actor_recall",
            "mean_context_recall",
            "mean_anchor_occupancy_iou",
            "mean_risk_fidelity_score",
            "perfect_case_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in list(comparison.get("profiles") or []):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _evaluate_case(case: Mapping[str, Any], prediction_index: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> Dict[str, Any]:
    benchmark_group = str(case.get("benchmark_group") or "")
    pred_by_sample_idx = dict(prediction_index.get(benchmark_group) or {})
    frame_metrics = []
    for frame in list(case.get("frames") or []):
        sample_idx = _safe_int(frame.get("sample_idx"))
        prediction = dict(pred_by_sample_idx.get(sample_idx) or {})
        gt_occupied = _cell_set(frame, "occupied_cells")
        pred_occupied = _cell_set(prediction, "occupied_cells")
        primary_cells = _cell_set(frame, "primary_actor_cells")
        context_cells = _cell_set(frame, "context_cells")
        frame_metrics.append(
            {
                "sample_idx": sample_idx,
                "occupancy_iou": _iou(gt_occupied, pred_occupied),
                "primary_actor_recall": _recall(primary_cells, pred_occupied),
                "context_recall": _recall(context_cells, pred_occupied),
                "has_prediction": bool(prediction),
            }
        )

    occupancy_values = [float(row["occupancy_iou"]) for row in frame_metrics]
    primary_values = [float(row["primary_actor_recall"]) for row in frame_metrics]
    context_values = [float(row["context_recall"]) for row in frame_metrics]
    anchor_sample_idx = _safe_int(case.get("anchor_sample_idx"))
    anchor_match = next((row for row in frame_metrics if _safe_int(row["sample_idx"]) == anchor_sample_idx), None)
    mean_occupancy_iou = mean(occupancy_values) if occupancy_values else 0.0
    mean_primary_recall = mean(primary_values) if primary_values else 0.0
    mean_context_recall = mean(context_values) if context_values else 0.0
    risk_fidelity = 0.45 * mean_occupancy_iou + 0.35 * mean_primary_recall + 0.20 * mean_context_recall
    failure_tags = []
    if not any(bool(row["has_prediction"]) for row in frame_metrics):
        failure_tags.append("missing_prediction")
    if mean_occupancy_iou < 0.55:
        failure_tags.append("low_occupancy_iou")
    if mean_primary_recall < 0.99:
        failure_tags.append("missed_primary_actor")
    if mean_context_recall < 0.50:
        failure_tags.append("context_undercoverage")

    return {
        "benchmark_group": benchmark_group,
        "reference_case_key": str(case.get("reference_case_key") or ""),
        "primary_behavior": str(case.get("primary_behavior") or ""),
        "category_group": str(case.get("category_group") or ""),
        "location": str(case.get("location") or ""),
        "frame_count": len(frame_metrics),
        "predicted_frame_count": sum(1 for row in frame_metrics if bool(row["has_prediction"])),
        "mean_occupancy_iou": round(mean_occupancy_iou, 4),
        "mean_primary_actor_recall": round(mean_primary_recall, 4),
        "mean_context_recall": round(mean_context_recall, 4),
        "anchor_occupancy_iou": round(float(anchor_match["occupancy_iou"]), 4) if anchor_match else 0.0,
        "risk_fidelity_score": round(risk_fidelity, 4),
        "failure_tags": failure_tags,
    }


def _build_summary(
    *,
    profile_name: str,
    benchmark_path: Path,
    predictions_path: Path,
    case_metrics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    occupancy_values = [float(row["mean_occupancy_iou"]) for row in case_metrics]
    primary_values = [float(row["mean_primary_actor_recall"]) for row in case_metrics]
    context_values = [float(row["mean_context_recall"]) for row in case_metrics]
    anchor_values = [float(row["anchor_occupancy_iou"]) for row in case_metrics]
    risk_values = [float(row["risk_fidelity_score"]) for row in case_metrics]
    return {
        "profile_name": profile_name,
        "benchmark_path": str(benchmark_path),
        "predictions_path": str(predictions_path),
        "overview": {
            "case_count": len(case_metrics),
            "mean_occupancy_iou": round(mean(occupancy_values), 4) if occupancy_values else 0.0,
            "mean_primary_actor_recall": round(mean(primary_values), 4) if primary_values else 0.0,
            "mean_context_recall": round(mean(context_values), 4) if context_values else 0.0,
            "mean_anchor_occupancy_iou": round(mean(anchor_values), 4) if anchor_values else 0.0,
            "mean_risk_fidelity_score": round(mean(risk_values), 4) if risk_values else 0.0,
            "perfect_case_count": sum(1 for row in case_metrics if not row.get("failure_tags")),
        },
        "behavior_breakdown": _group_breakdown(case_metrics, "primary_behavior"),
        "risk_breakdowns": {field: _group_breakdown(case_metrics, field) for field in RISK_FACET_FIELDS},
        "case_metrics": list(case_metrics),
    }


def _group_breakdown(rows: Sequence[Mapping[str, Any]], field_name: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field_name) or "unknown")].append(row)
    output = []
    for value, group_rows in sorted(grouped.items()):
        failures = Counter(
            tag
            for row in group_rows
            for tag in list(row.get("failure_tags") or [])
        )
        output.append(
            {
                field_name: value,
                "case_count": len(group_rows),
                "mean_occupancy_iou": round(mean(float(row["mean_occupancy_iou"]) for row in group_rows), 4),
                "mean_primary_actor_recall": round(mean(float(row["mean_primary_actor_recall"]) for row in group_rows), 4),
                "mean_context_recall": round(mean(float(row["mean_context_recall"]) for row in group_rows), 4),
                "mean_risk_fidelity_score": round(mean(float(row["risk_fidelity_score"]) for row in group_rows), 4),
                "top_failure_summary": _failure_summary(failures),
            }
        )
    return output


def _render_markdown(summary: Mapping[str, Any]) -> str:
    overview = dict(summary.get("overview") or {})
    lines = [
        "# BEV Occupancy Slice Evaluation",
        "",
        f"- Profile: {summary.get('profile_name', '')}",
        f"- Cases: {overview.get('case_count', 0)}",
        f"- Mean occupancy IoU: {float(overview.get('mean_occupancy_iou') or 0.0):.3f}",
        f"- Mean primary actor recall: {float(overview.get('mean_primary_actor_recall') or 0.0):.3f}",
        f"- Mean context recall: {float(overview.get('mean_context_recall') or 0.0):.3f}",
        f"- Mean risk fidelity: {float(overview.get('mean_risk_fidelity_score') or 0.0):.3f}",
        "",
        "## Behavior Breakdown",
        "",
        "| Behavior | Cases | Occupancy IoU | Primary Recall | Context Recall | Risk Fidelity | Top Failure Modes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(summary.get("behavior_breakdown") or []):
        lines.append(
            "| {0} | {1} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6} |".format(
                row["primary_behavior"],
                int(row["case_count"]),
                float(row["mean_occupancy_iou"]),
                float(row["mean_primary_actor_recall"]),
                float(row["mean_context_recall"]),
                float(row["mean_risk_fidelity_score"]),
                row["top_failure_summary"],
            )
        )
    lines.extend(
        [
            "",
            "## Case Metrics",
            "",
            "| Group | Behavior | Occupancy IoU | Primary Recall | Context Recall | Anchor IoU | Risk Fidelity | Failure Tags |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in list(summary.get("case_metrics") or []):
        lines.append(
            "| {0} | {1} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6:.3f} | {7} |".format(
                row["benchmark_group"],
                row["primary_behavior"],
                float(row["mean_occupancy_iou"]),
                float(row["mean_primary_actor_recall"]),
                float(row["mean_context_recall"]),
                float(row["anchor_occupancy_iou"]),
                float(row["risk_fidelity_score"]),
                ", ".join(str(tag) for tag in list(row.get("failure_tags") or [])) or "none",
            )
        )
    return "\n".join(lines) + "\n"


def _write_case_metrics_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "benchmark_group",
        "reference_case_key",
        "primary_behavior",
        "category_group",
        "location",
        "frame_count",
        "predicted_frame_count",
        "mean_occupancy_iou",
        "mean_primary_actor_recall",
        "mean_context_recall",
        "anchor_occupancy_iou",
        "risk_fidelity_score",
        *RISK_FACET_FIELDS,
        "min_distance_m",
        "min_ttc_s",
        "failure_tags",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["failure_tags"] = "|".join(str(tag) for tag in list(row.get("failure_tags") or []))
            writer.writerow({field: payload.get(field, "") for field in fieldnames})


def _build_prediction_index(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[int, Mapping[str, Any]]]:
    index: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in predictions:
        group = str(record.get("benchmark_group") or "")
        for frame in list(record.get("frames") or []):
            index[group][_safe_int(frame.get("sample_idx"))] = dict(frame)
    return index


def _load_agent_context(conn: sqlite3.Connection, sample_tokens: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not sample_tokens:
        return {}
    placeholders = ", ".join("?" for _ in sample_tokens)
    rows = conn.execute(
        """
        SELECT sample_token, sample_idx, instance_token, category_group, x_ego, y_ego
        FROM agents
        WHERE sample_token IN ({0})
        """.format(placeholders),
        tuple(sample_tokens),
    ).fetchall()
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_token"])].append(
            {
                "sample_token": str(row["sample_token"]),
                "sample_idx": _safe_int(row["sample_idx"]),
                "instance_token": str(row["instance_token"]),
                "category_group": _normalize_category(str(row["category_group"])),
                "x_ego": _safe_float(row["x_ego"]),
                "y_ego": _safe_float(row["y_ego"]),
            }
        )
    return by_sample


def _normalize_grid_spec(grid_spec: Optional[Mapping[str, float]]) -> Dict[str, float]:
    grid = dict(DEFAULT_BEV_GRID_SPEC)
    if grid_spec:
        grid.update({key: float(value) for key, value in grid_spec.items()})
    return grid


def _rasterize_points(points: Sequence[Tuple[float, float]], grid_spec: Mapping[str, float]) -> List[List[int]]:
    cells = []
    for x_ego, y_ego in points:
        cell = _cell_for_point(float(x_ego), float(y_ego), grid_spec)
        if cell is not None:
            cells.append(cell)
    return _sorted_cells(_dilate_cells(cells, int(float(grid_spec.get("dilation_radius_cells", 1.0)))))


def _cell_for_point(x_ego: float, y_ego: float, grid_spec: Mapping[str, float]) -> Optional[Tuple[int, int]]:
    x_min = float(grid_spec["x_min_m"])
    x_max = float(grid_spec["x_max_m"])
    y_min = float(grid_spec["y_min_m"])
    y_max = float(grid_spec["y_max_m"])
    resolution = float(grid_spec["resolution_m"])
    if not (x_min <= x_ego < x_max and y_min <= y_ego < y_max):
        return None
    return int(math.floor((x_ego - x_min) / resolution)), int(math.floor((y_ego - y_min) / resolution))


def _dilate_cells(cells: Iterable[Tuple[int, int]], radius_cells: int) -> set[Tuple[int, int]]:
    expanded: set[Tuple[int, int]] = set()
    for x_idx, y_idx in cells:
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                expanded.add((int(x_idx + dx), int(y_idx + dy)))
    return expanded


def _cell_set(row: Mapping[str, Any], key: str) -> set[Tuple[int, int]]:
    return {
        (int(cell[0]), int(cell[1]))
        for cell in list(row.get(key) or [])
        if isinstance(cell, (list, tuple)) and len(cell) >= 2
    }


def _union_cells(*cell_groups: Sequence[Sequence[int]]) -> List[List[int]]:
    cells = []
    for group in cell_groups:
        cells.extend((int(cell[0]), int(cell[1])) for cell in list(group or []) if len(cell) >= 2)
    return _sorted_cells(cells)


def _sorted_cells(cells: Iterable[Tuple[int, int]]) -> List[List[int]]:
    return [[int(x), int(y)] for x, y in sorted(set(cells), key=lambda item: (item[0], item[1]))]


def _iou(lhs: set[Tuple[int, int]], rhs: set[Tuple[int, int]]) -> float:
    if not lhs and not rhs:
        return 1.0
    union = lhs | rhs
    if not union:
        return 0.0
    return float(len(lhs & rhs)) / float(len(union))


def _recall(target: set[Tuple[int, int]], predicted: set[Tuple[int, int]]) -> float:
    if not target:
        return 1.0
    return float(len(target & predicted)) / float(len(target))


def _case_distribution(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    by_behavior = Counter(str(case.get("primary_behavior") or "unknown") for case in cases)
    by_location = Counter(str(case.get("location") or "unknown") for case in cases)
    return {
        "by_behavior": dict(sorted(by_behavior.items())),
        "by_location": dict(sorted(by_location.items())),
    }


def _failure_summary(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{tag}:{count}" for tag, count in counter.most_common(3))


def _normalize_category(value: str) -> str:
    key = str(value or "").strip().lower()
    return CATEGORY_ALIASES.get(key, key)


def _unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _markdown_to_basic_html(markdown: str) -> str:
    return "<!doctype html><html><body><pre>" + html.escape(markdown) + "</pre></body></html>"
