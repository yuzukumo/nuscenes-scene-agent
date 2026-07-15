from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence

from jinja2 import Template

from nusc_scene_agent.score_semantics import (
    get_average_best_validation_quality_score,
    get_best_validation_quality_score,
    get_max_validation_quality_score,
)


SUMMARY_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>nuScenes Benchmark Metrics</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 32px; background: #f5f1ea; color: #1f1f1f; }
    h1, h2 { margin-bottom: 0.4rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; background: white; }
    th, td { border-bottom: 1px solid #ddd4c7; padding: 10px 12px; text-align: left; }
    th { background: #ede5d8; }
    .meta { color: #665d54; margin-bottom: 18px; }
  </style>
</head>
<body>
  <h1>nuScenes Benchmark Metrics</h1>
  <div class="meta">
    Queries: {{ metrics.overview.query_count }} |
    Validation acceptance@1: {{ metrics.overview.validation_acceptance_at_1_count }}/{{ metrics.overview.query_count }} ({{ "%.1f"|format(metrics.overview.validation_acceptance_at_1_rate * 100.0) }}%) |
    Final selection acceptance@1: {{ metrics.overview.final_selection_acceptance_at_1_count }}/{{ metrics.overview.query_count }} ({{ "%.1f"|format(metrics.overview.final_selection_acceptance_at_1_rate * 100.0) }}%)
  </div>

  <h2>Overview</h2>
  <table>
    <tbody>
      <tr><th>Selected Cases</th><td>{{ metrics.overview.selected_case_count }}</td></tr>
      <tr><th>Passed Selected Cases</th><td>{{ metrics.overview.passed_selected_case_count }}</td></tr>
      <tr><th>Unique Cases</th><td>{{ metrics.overview.unique_case_count }}</td></tr>
      <tr><th>Unique Passed Cases</th><td>{{ metrics.overview.unique_passed_case_count }}</td></tr>
      <tr><th>Unique Locations</th><td>{{ metrics.overview.unique_locations }}</td></tr>
      <tr><th>Mean Best Validation Quality</th><td>{{ "%.2f"|format(metrics.overview.mean_best_validation_quality_score) }}</td></tr>
      <tr><th>Median Best Validation Quality</th><td>{{ "%.2f"|format(metrics.overview.median_best_validation_quality_score) }}</td></tr>
      <tr><th>Mean Max Quality (ungated diagnostic)</th><td>{{ "%.2f"|format(metrics.overview.mean_max_validation_quality_score) }}</td></tr>
    </tbody>
  </table>

  {% if metrics.ranking_stage_metrics is defined %}
  <h2>Ranking Stages</h2>
  <table>
    <thead>
      <tr><th>Stage</th><th>Reference objective@1</th><th>Reference objective@K</th><th>Mean candidates</th></tr>
    </thead>
    <tbody>
    {% for stage_name, stage in metrics.ranking_stage_metrics.items() %}
      <tr>
        <td>{{ stage_name }}</td>
        <td>{{ stage.reference_objective_at_1_count }}/{{ stage.query_count }} ({{ "%.1f"|format(stage.reference_objective_at_1_rate * 100.0) }}%)</td>
        <td>{{ stage.reference_objective_at_k_count }}/{{ stage.query_count }} ({{ "%.1f"|format(stage.reference_objective_at_k_rate * 100.0) }}%)</td>
        <td>{{ "%.2f"|format(stage.mean_candidate_count) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}

  {% if metrics.reference_metrics is defined and metrics.reference_metrics.query_count > 0 %}
  <h2>Final-Selection Reference Metrics</h2>
  <table>
    <tbody>
      <tr><th>Labeled Queries</th><td>{{ metrics.reference_metrics.query_count }}</td></tr>
      <tr><th>Scene Objective@1</th><td>{{ metrics.reference_metrics.scene_objective_at_1_count }}/{{ metrics.reference_metrics.query_count }} ({{ "%.1f"|format(metrics.reference_metrics.scene_objective_at_1_rate * 100.0) }}%)</td></tr>
      <tr><th>Scene Objective@K</th><td>{{ metrics.reference_metrics.scene_objective_at_k_count }}/{{ metrics.reference_metrics.query_count }} ({{ "%.1f"|format(metrics.reference_metrics.scene_objective_at_k_rate * 100.0) }}%)</td></tr>
      <tr><th>Actor Objective@1</th><td>{{ metrics.reference_metrics.actor_objective_at_1_count }}/{{ metrics.reference_metrics.query_count }} ({{ "%.1f"|format(metrics.reference_metrics.actor_objective_at_1_rate * 100.0) }}%)</td></tr>
      <tr><th>Actor Objective@K</th><td>{{ metrics.reference_metrics.actor_objective_at_k_count }}/{{ metrics.reference_metrics.query_count }} ({{ "%.1f"|format(metrics.reference_metrics.actor_objective_at_k_rate * 100.0) }}%)</td></tr>
      <tr><th>Reference Objective@1</th><td>{{ metrics.reference_metrics.objective_at_1_count }}/{{ metrics.reference_metrics.query_count }} ({{ "%.1f"|format(metrics.reference_metrics.objective_at_1_rate * 100.0) }}%)</td></tr>
      <tr><th>Reference Objective@K</th><td>{{ metrics.reference_metrics.objective_at_k_count }}/{{ metrics.reference_metrics.query_count }} ({{ "%.1f"|format(metrics.reference_metrics.objective_at_k_rate * 100.0) }}%)</td></tr>
      <tr><th>Positive Localization Queries</th><td>{{ metrics.reference_metrics.positive_localization_query_count }}</td></tr>
      <tr><th>Mean Event IoU</th><td>{{ "%.3f"|format(metrics.reference_metrics.mean_event_iou) }}</td></tr>
      <tr><th>Mean Peak Error</th><td>{{ "%.2f"|format(metrics.reference_metrics.mean_peak_error) }}</td></tr>
      <tr><th>Contrastive Groups</th><td>{{ metrics.reference_metrics.contrastive_group_count }}</td></tr>
      <tr><th>Contrastive Success@1</th><td>{{ metrics.reference_metrics.contrastive_group_success_at_1_count }}/{{ metrics.reference_metrics.contrastive_group_count }} ({{ "%.1f"|format(metrics.reference_metrics.contrastive_group_success_at_1_rate * 100.0) }}%)</td></tr>
      <tr><th>Contrastive Success@K</th><td>{{ metrics.reference_metrics.contrastive_group_success_at_k_count }}/{{ metrics.reference_metrics.contrastive_group_count }} ({{ "%.1f"|format(metrics.reference_metrics.contrastive_group_success_at_k_rate * 100.0) }}%)</td></tr>
    </tbody>
  </table>
  {% endif %}

  <h2>Query Breakdown</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Raw Ref@1</th><th>Reranked Ref@1</th><th>Validation Accept@1</th><th>Final Accept@1</th><th>Final Ref@1</th><th>Final Ref@K</th><th>Passed</th><th>Selected</th><th>Best Validation Quality</th><th>Actors</th><th>Behaviors</th>
      </tr>
    </thead>
    <tbody>
    {% for row in metrics.query_metrics %}
      <tr>
        <td>{{ row.id }}</td>
        <td>{{ row.ranking_stages.raw_retrieval.reference_objective_at_1 if row.ranking_stages.raw_retrieval.reference_objective_at_1 is not none else "-" }}</td>
        <td>{{ row.ranking_stages.reranked.reference_objective_at_1 if row.ranking_stages.reranked.reference_objective_at_1 is not none else "-" }}</td>
        <td>{{ row.validation_acceptance_at_1 }}</td>
        <td>{{ row.final_selection_acceptance_at_1 }}</td>
        <td>{{ row.reference_objective_at_1 if row.reference_objective_at_1 is not none else "-" }}</td>
        <td>{{ row.reference_objective_at_k if row.reference_objective_at_k is not none else "-" }}</td>
        <td>{{ row.passed_count }}</td>
        <td>{{ row.selected_count }}</td>
        <td>{{ "%.2f"|format(row.best_validation_quality_score) }}</td>
        <td>{{ ", ".join(row.actors) }}</td>
        <td>{{ ", ".join(row.behaviors) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Behavior Coverage</h2>
  <table>
    <thead>
      <tr>
        <th>Behavior</th><th>Queries</th><th>Final Accept@1</th><th>Final Accept@K</th><th>Avg Best Validation Quality</th>
      </tr>
    </thead>
    <tbody>
    {% for row in metrics.behavior_coverage %}
      <tr>
        <td>{{ row.name }}</td>
        <td>{{ row.query_count }}</td>
        <td>{{ row.pass_at_1_count }}</td>
        <td>{{ row.pass_at_k_count }}</td>
        <td>{{ "%.2f"|format(row.avg_best_validation_quality_score) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Actor Coverage</h2>
  <table>
    <thead>
      <tr>
        <th>Actor</th><th>Queries</th><th>Final Accept@1</th><th>Final Accept@K</th><th>Avg Best Validation Quality</th>
      </tr>
    </thead>
    <tbody>
    {% for row in metrics.actor_coverage %}
      <tr>
        <td>{{ row.name }}</td>
        <td>{{ row.query_count }}</td>
        <td>{{ row.pass_at_1_count }}</td>
        <td>{{ row.pass_at_k_count }}</td>
        <td>{{ "%.2f"|format(row.avg_best_validation_quality_score) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Location Distribution</h2>
  <table>
    <thead>
      <tr>
        <th>Location</th><th>Cases</th><th>Passed</th>
      </tr>
    </thead>
    <tbody>
    {% for row in metrics.location_distribution %}
      <tr>
        <td>{{ row.location }}</td>
        <td>{{ row.case_count }}</td>
        <td>{{ row.passed_case_count }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""
)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _group_coverage(query_metrics: Sequence[Dict[str, object]], field_name: str, empty_label: str) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "query_count": 0,
            "pass_at_1_count": 0,
            "pass_at_k_count": 0,
            "best_scores": [],
        }
    )

    for row in query_metrics:
        items = list(row.get(field_name) or [])
        if not items:
            items = [empty_label]
        for item in items:
            group = grouped[str(item)]
            group["query_count"] += 1
            group["pass_at_1_count"] += int(bool(row["pass_at_1"]))
            group["pass_at_k_count"] += int(bool(row["pass_at_k"]))
            group["best_scores"].append(get_best_validation_quality_score(row))

    coverage_rows: List[Dict[str, object]] = []
    for name, stats in grouped.items():
        query_count = int(stats["query_count"])
        pass_at_1_count = int(stats["pass_at_1_count"])
        pass_at_k_count = int(stats["pass_at_k_count"])
        best_scores = [float(item) for item in stats["best_scores"]]
        coverage_rows.append(
            {
                "name": name,
                "query_count": query_count,
                "pass_at_1_count": pass_at_1_count,
                "pass_at_k_count": pass_at_k_count,
                "pass_at_1_rate": _ratio(pass_at_1_count, query_count),
                "pass_at_k_rate": _ratio(pass_at_k_count, query_count),
                "avg_best_validation_quality_score": round(mean(best_scores), 2) if best_scores else 0.0,
            }
        )

    coverage_rows.sort(
        key=lambda item: (
            int(item["pass_at_k_count"]),
            float(item["avg_best_validation_quality_score"]),
            -len(str(item["name"])),
        ),
        reverse=True,
    )
    return coverage_rows


def _selected_case_key(case: object) -> str:
    return "{0}:{1}".format(case.candidate.sample_token, case.candidate.instance_token)


def _candidate_from_ranked_item(item: object) -> object:
    return getattr(item, "candidate", item)


def _rank_stage_metrics(
    items: Sequence[object],
    *,
    reference_case_keys: Sequence[str],
    reference_scene_names: Sequence[str],
    reference_instance_tokens: Sequence[str],
    expect_match: object,
) -> Dict[str, object]:
    candidates = [_candidate_from_ranked_item(item) for item in items]
    case_keys = ["{0}:{1}".format(item.sample_token, item.instance_token) for item in candidates]
    scene_names = [str(item.scene_name) for item in candidates]
    instance_tokens = [str(item.instance_token) for item in candidates]

    scene_hit_at_1 = bool(scene_names and scene_names[0] in reference_scene_names)
    scene_hit_at_k = any(name in reference_scene_names for name in scene_names)
    actor_hit_at_1 = bool(instance_tokens and instance_tokens[0] in reference_instance_tokens)
    actor_hit_at_k = any(token in reference_instance_tokens for token in instance_tokens)
    reference_hit_at_1 = bool(case_keys and case_keys[0] in reference_case_keys)
    reference_hit_at_k = any(case_key in reference_case_keys for case_key in case_keys)

    if expect_match is None or not reference_case_keys:
        scene_objective_at_1 = None
        scene_objective_at_k = None
        actor_objective_at_1 = None
        actor_objective_at_k = None
        reference_objective_at_1 = None
        reference_objective_at_k = None
    else:
        positive = bool(expect_match)
        scene_objective_at_1 = scene_hit_at_1 if positive else not scene_hit_at_1
        scene_objective_at_k = scene_hit_at_k if positive else not scene_hit_at_k
        actor_objective_at_1 = actor_hit_at_1 if positive else not actor_hit_at_1
        actor_objective_at_k = actor_hit_at_k if positive else not actor_hit_at_k
        reference_objective_at_1 = reference_hit_at_1 if positive else not reference_hit_at_1
        reference_objective_at_k = reference_hit_at_k if positive else not reference_hit_at_k

    return {
        "candidate_count": len(candidates),
        "scene_hit_at_1": scene_hit_at_1,
        "scene_hit_at_k": scene_hit_at_k,
        "actor_hit_at_1": actor_hit_at_1,
        "actor_hit_at_k": actor_hit_at_k,
        "reference_hit_at_1": reference_hit_at_1,
        "reference_hit_at_k": reference_hit_at_k,
        "scene_objective_at_1": scene_objective_at_1,
        "scene_objective_at_k": scene_objective_at_k,
        "actor_objective_at_1": actor_objective_at_1,
        "actor_objective_at_k": actor_objective_at_k,
        "reference_objective_at_1": reference_objective_at_1,
        "reference_objective_at_k": reference_objective_at_k,
        "top_case_key": case_keys[0] if case_keys else "",
    }


def _aggregate_rank_stage(query_metrics: Sequence[Dict[str, object]], stage_name: str) -> Dict[str, object]:
    rows = [
        row
        for row in query_metrics
        if row.get("expect_match") is not None and row.get("reference_case_keys")
    ]
    fields = [
        "scene_objective_at_1",
        "scene_objective_at_k",
        "actor_objective_at_1",
        "actor_objective_at_k",
        "reference_objective_at_1",
        "reference_objective_at_k",
    ]
    payload: Dict[str, object] = {
        "query_count": len(rows),
        "mean_candidate_count": round(
            mean(float(dict(row.get("ranking_stages") or {}).get(stage_name, {}).get("candidate_count") or 0) for row in rows),
            2,
        )
        if rows
        else 0.0,
    }
    for field in fields:
        count = sum(
            1
            for row in rows
            if bool(dict(row.get("ranking_stages") or {}).get(stage_name, {}).get(field))
        )
        payload[field + "_count"] = count
        payload[field + "_rate"] = round(_ratio(count, len(rows)), 4)
    return payload


def _event_range_iou(predicted_range: Sequence[int], reference_range: Sequence[int]) -> float:
    if len(predicted_range) != 2 or len(reference_range) != 2:
        return 0.0
    p0, p1 = int(predicted_range[0]), int(predicted_range[1])
    r0, r1 = int(reference_range[0]), int(reference_range[1])
    inter = max(0, min(p1, r1) - max(p0, r0) + 1)
    union = max(p1, r1) - min(p0, r0) + 1
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _reference_metrics(query_metrics: Sequence[Dict[str, object]]) -> Dict[str, object]:
    labeled = [row for row in query_metrics if row.get("expect_match") is not None and row.get("reference_case_keys")]
    if not labeled:
        return {
            "query_count": 0,
            "scene_objective_at_1_count": 0,
            "scene_objective_at_1_rate": 0.0,
            "scene_objective_at_k_count": 0,
            "scene_objective_at_k_rate": 0.0,
            "actor_objective_at_1_count": 0,
            "actor_objective_at_1_rate": 0.0,
            "actor_objective_at_k_count": 0,
            "actor_objective_at_k_rate": 0.0,
            "objective_at_1_count": 0,
            "objective_at_1_rate": 0.0,
            "objective_at_k_count": 0,
            "objective_at_k_rate": 0.0,
            "positive_localization_query_count": 0,
            "mean_event_iou": 0.0,
            "mean_peak_error": 0.0,
            "contrastive_group_count": 0,
            "contrastive_group_success_at_1_count": 0,
            "contrastive_group_success_at_1_rate": 0.0,
            "contrastive_group_success_at_k_count": 0,
            "contrastive_group_success_at_k_rate": 0.0,
        }

    by_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in labeled:
        group = str(row.get("benchmark_group") or "")
        if group:
            by_group[group].append(row)

    objective_at_1_count = sum(1 for row in labeled if row["reference_objective_at_1"])
    objective_at_k_count = sum(1 for row in labeled if row["reference_objective_at_k"])
    scene_objective_at_1_count = sum(1 for row in labeled if row["scene_objective_at_1"])
    scene_objective_at_k_count = sum(1 for row in labeled if row["scene_objective_at_k"])
    actor_objective_at_1_count = sum(1 for row in labeled if row["actor_objective_at_1"])
    actor_objective_at_k_count = sum(1 for row in labeled if row["actor_objective_at_k"])
    localization_rows = [
        row
        for row in labeled
        if row.get("expect_match") is True and row.get("actor_objective_at_k") and row.get("event_iou") is not None
    ]
    group_success_at_1_count = sum(1 for rows in by_group.values() if all(item["reference_objective_at_1"] for item in rows))
    group_success_at_k_count = sum(1 for rows in by_group.values() if all(item["reference_objective_at_k"] for item in rows))

    return {
        "query_count": len(labeled),
        "scene_objective_at_1_count": scene_objective_at_1_count,
        "scene_objective_at_1_rate": round(_ratio(scene_objective_at_1_count, len(labeled)), 4),
        "scene_objective_at_k_count": scene_objective_at_k_count,
        "scene_objective_at_k_rate": round(_ratio(scene_objective_at_k_count, len(labeled)), 4),
        "actor_objective_at_1_count": actor_objective_at_1_count,
        "actor_objective_at_1_rate": round(_ratio(actor_objective_at_1_count, len(labeled)), 4),
        "actor_objective_at_k_count": actor_objective_at_k_count,
        "actor_objective_at_k_rate": round(_ratio(actor_objective_at_k_count, len(labeled)), 4),
        "objective_at_1_count": objective_at_1_count,
        "objective_at_1_rate": round(_ratio(objective_at_1_count, len(labeled)), 4),
        "objective_at_k_count": objective_at_k_count,
        "objective_at_k_rate": round(_ratio(objective_at_k_count, len(labeled)), 4),
        "positive_localization_query_count": len(localization_rows),
        "mean_event_iou": round(mean(float(row["event_iou"]) for row in localization_rows), 4) if localization_rows else 0.0,
        "mean_peak_error": round(mean(float(row["peak_error"]) for row in localization_rows if row.get("peak_error") is not None), 4)
        if localization_rows and any(row.get("peak_error") is not None for row in localization_rows)
        else 0.0,
        "contrastive_group_count": len(by_group),
        "contrastive_group_success_at_1_count": group_success_at_1_count,
        "contrastive_group_success_at_1_rate": round(_ratio(group_success_at_1_count, len(by_group)), 4),
        "contrastive_group_success_at_k_count": group_success_at_k_count,
        "contrastive_group_success_at_k_rate": round(_ratio(group_success_at_k_count, len(by_group)), 4),
    }


def build_benchmark_metrics(
    benchmark_results: Sequence[Dict[str, object]],
    case_library_entries: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    query_metrics: List[Dict[str, object]] = []

    for result in benchmark_results:
        query_spec = result.get("query_spec")
        selected_cases = list(result.get("selected_cases") or [])
        retrieval_candidates = list(
            result.get("retrieval_candidates")
            or result.get("candidates")
            or [case.candidate for case in selected_cases]
        )
        ranked_candidates = list(
            result.get("ranked_candidates")
            or result.get("candidates")
            or retrieval_candidates
        )
        validated_cases = list(result.get("validated_cases") or selected_cases)
        passed_count = sum(1 for case in selected_cases if case.passed)
        best_case = max(
            selected_cases,
            key=lambda item: (
                bool(item.passed),
                float(item.validation_quality_score),
            ),
            default=None,
        )
        best_quality_case = max(
            selected_cases,
            key=lambda item: float(item.validation_quality_score),
            default=None,
        )
        reference_case_keys = list(getattr(query_spec, "reference_case_keys", []))
        reference_scene_names = list(getattr(query_spec, "reference_scene_names", []))
        reference_instance_tokens = list(getattr(query_spec, "reference_instance_tokens", []))
        reference_event_sample_range = list(getattr(query_spec, "reference_event_sample_range", []))
        reference_peak_sample_idx = getattr(query_spec, "reference_peak_sample_idx", None)
        expect_match = getattr(query_spec, "expect_match", None)
        stage_kwargs = {
            "reference_case_keys": reference_case_keys,
            "reference_scene_names": reference_scene_names,
            "reference_instance_tokens": reference_instance_tokens,
            "expect_match": expect_match,
        }
        ranking_stages = {
            "raw_retrieval": _rank_stage_metrics(retrieval_candidates, **stage_kwargs),
            "reranked": _rank_stage_metrics(ranked_candidates, **stage_kwargs),
            "validation_order": _rank_stage_metrics(validated_cases, **stage_kwargs),
            "final_selection": _rank_stage_metrics(selected_cases, **stage_kwargs),
        }
        final_rank = ranking_stages["final_selection"]
        scene_hit_at_1 = bool(final_rank["scene_hit_at_1"])
        scene_hit_at_k = bool(final_rank["scene_hit_at_k"])
        actor_hit_at_1 = bool(final_rank["actor_hit_at_1"])
        actor_hit_at_k = bool(final_rank["actor_hit_at_k"])
        reference_hit_at_1 = bool(final_rank["reference_hit_at_1"])
        reference_hit_at_k = bool(final_rank["reference_hit_at_k"])
        scene_objective_at_1 = final_rank["scene_objective_at_1"]
        scene_objective_at_k = final_rank["scene_objective_at_k"]
        actor_objective_at_1 = final_rank["actor_objective_at_1"]
        actor_objective_at_k = final_rank["actor_objective_at_k"]
        reference_objective_at_1 = final_rank["reference_objective_at_1"]
        reference_objective_at_k = final_rank["reference_objective_at_k"]
        top_case = selected_cases[0] if selected_cases else None
        predicted_event_range = []
        predicted_peak_sample_idx = None
        if top_case is not None and top_case.event_localization:
            predicted_event_range = [
                int(top_case.event_localization.get("start_sample_idx")),
                int(top_case.event_localization.get("end_sample_idx")),
            ]
            predicted_peak_sample_idx = top_case.event_localization.get("peak_sample_idx")
        event_iou = (
            round(_event_range_iou(predicted_event_range, reference_event_sample_range), 4)
            if len(reference_event_sample_range) == 2 and len(predicted_event_range) == 2
            else None
        )
        peak_error = (
            abs(int(predicted_peak_sample_idx) - int(reference_peak_sample_idx))
            if predicted_peak_sample_idx is not None and reference_peak_sample_idx is not None
            else None
        )

        query_metrics.append(
            {
                "id": str(result.get("id") or getattr(result.get("query"), "original_text", "query")),
                "description": str(getattr(query_spec, "description", "") or ""),
                "query_text": str(result["query"].original_text),
                "tags": list(getattr(query_spec, "tags", [])),
                "actors": list(getattr(query_spec, "actors", [])),
                "behaviors": list(getattr(query_spec, "behaviors", [])),
                "resolved_category_groups": list(result["query"].category_groups),
                "resolved_positions": list(result["query"].positions),
                "resolved_behaviors": list(result["query"].behaviors),
                "resolved_risk_terms": list(result["query"].risk_terms),
                "reference_case_keys": reference_case_keys,
                "reference_scene_names": reference_scene_names,
                "reference_instance_tokens": reference_instance_tokens,
                "reference_event_sample_range": reference_event_sample_range,
                "reference_peak_sample_idx": reference_peak_sample_idx,
                "expect_match": expect_match,
                "benchmark_group": str(getattr(query_spec, "benchmark_group", "") or ""),
                "variant_type": str(getattr(query_spec, "variant_type", "") or ""),
                "ranking_stages": ranking_stages,
                "scene_hit_at_1": scene_hit_at_1,
                "scene_hit_at_k": scene_hit_at_k,
                "actor_hit_at_1": actor_hit_at_1,
                "actor_hit_at_k": actor_hit_at_k,
                "scene_objective_at_1": scene_objective_at_1,
                "scene_objective_at_k": scene_objective_at_k,
                "actor_objective_at_1": actor_objective_at_1,
                "actor_objective_at_k": actor_objective_at_k,
                "reference_hit_at_1": reference_hit_at_1,
                "reference_hit_at_k": reference_hit_at_k,
                "reference_objective_at_1": reference_objective_at_1,
                "reference_objective_at_k": reference_objective_at_k,
                "event_iou": event_iou,
                "peak_error": peak_error,
                "selected_count": len(selected_cases),
                "retrieval_candidate_count": len(retrieval_candidates),
                "ranked_candidate_count": len(ranked_candidates),
                "validated_candidate_count": len(validated_cases),
                "passed_count": passed_count,
                "validation_acceptance_at_1": bool(validated_cases and validated_cases[0].passed),
                "validation_acceptance_at_k": bool(any(case.passed for case in validated_cases)),
                "final_selection_acceptance_at_1": bool(selected_cases and selected_cases[0].passed),
                "final_selection_acceptance_at_k": bool(passed_count > 0),
                # Backward-compatible aliases. These are final-selection acceptance,
                # not raw retrieval metrics.
                "pass_at_1": bool(selected_cases and selected_cases[0].passed),
                "pass_at_k": bool(passed_count > 0),
                # Follow the selection policy used by the pipeline: accepted
                # cases take precedence, then quality is maximized.
                "best_validation_quality_score": (
                    float(best_case.validation_quality_score) if best_case is not None else 0.0
                ),
                # Keep the unconstrained maximum as a separate diagnostic. It
                # may belong to a rejected case and must not be confused with
                # the case selected for the benchmark library.
                "max_validation_quality_score": (
                    float(best_quality_case.validation_quality_score) if best_quality_case is not None else 0.0
                ),
                # Backward-compatible alias for artifacts produced before the
                # quality/gate distinction was introduced.
                "best_validation_score": float(best_case.validation_quality_score) if best_case is not None else 0.0,
                "best_scene_name": str(best_case.candidate.scene_name) if best_case is not None else "",
                "best_sample_idx": int(best_case.candidate.sample_idx) if best_case is not None else -1,
                "locations": sorted({str(case.candidate.location) for case in selected_cases}),
            }
        )

    query_count = len(query_metrics)
    pass_at_1_count = sum(1 for row in query_metrics if row["pass_at_1"])
    pass_at_k_count = sum(1 for row in query_metrics if row["pass_at_k"])
    validation_at_1_count = sum(1 for row in query_metrics if row["validation_acceptance_at_1"])
    validation_at_k_count = sum(1 for row in query_metrics if row["validation_acceptance_at_k"])
    best_scores = [get_best_validation_quality_score(row) for row in query_metrics]
    max_quality_scores = [get_max_validation_quality_score(row) for row in query_metrics]
    selected_case_count = sum(int(row["selected_count"]) for row in query_metrics)
    passed_selected_case_count = sum(int(row["passed_count"]) for row in query_metrics)
    location_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"case_count": 0, "passed_case_count": 0})

    for entry in case_library_entries:
        location = str(entry.get("location") or "unknown")
        location_counts[location]["case_count"] += 1
        location_counts[location]["passed_case_count"] += int(bool(entry.get("passed")))

    location_distribution = [
        {
            "location": location,
            "case_count": stats["case_count"],
            "passed_case_count": stats["passed_case_count"],
        }
        for location, stats in location_counts.items()
    ]
    location_distribution.sort(key=lambda item: (int(item["case_count"]), int(item["passed_case_count"])), reverse=True)

    reference_metrics = _reference_metrics(query_metrics)
    metrics = {
        "overview": {
            "query_count": query_count,
            "selected_case_count": selected_case_count,
            "passed_selected_case_count": passed_selected_case_count,
            "unique_case_count": len(case_library_entries),
            "unique_passed_case_count": sum(1 for entry in case_library_entries if entry.get("passed")),
            "pass_at_1_count": pass_at_1_count,
            "pass_at_1_rate": round(_ratio(pass_at_1_count, query_count), 4),
            "pass_at_k_count": pass_at_k_count,
            "pass_at_k_rate": round(_ratio(pass_at_k_count, query_count), 4),
            "final_selection_acceptance_at_1_count": pass_at_1_count,
            "final_selection_acceptance_at_1_rate": round(_ratio(pass_at_1_count, query_count), 4),
            "final_selection_acceptance_at_k_count": pass_at_k_count,
            "final_selection_acceptance_at_k_rate": round(_ratio(pass_at_k_count, query_count), 4),
            "validation_acceptance_at_1_count": validation_at_1_count,
            "validation_acceptance_at_1_rate": round(_ratio(validation_at_1_count, query_count), 4),
            "validation_acceptance_at_k_count": validation_at_k_count,
            "validation_acceptance_at_k_rate": round(_ratio(validation_at_k_count, query_count), 4),
            "unique_locations": len(location_distribution),
            "mean_best_validation_quality_score": (
                round(mean(best_scores), 2) if best_scores else 0.0
            ),
            "median_best_validation_quality_score": (
                round(median(best_scores), 2) if best_scores else 0.0
            ),
            "mean_max_validation_quality_score": (
                round(mean(max_quality_scores), 2) if max_quality_scores else 0.0
            ),
            "median_max_validation_quality_score": (
                round(median(max_quality_scores), 2) if max_quality_scores else 0.0
            ),
            # Backward-compatible aliases. Canonical consumers should use the
            # explicitly named quality fields above.
            "mean_best_validation_score": round(mean(best_scores), 2) if best_scores else 0.0,
            "median_best_validation_score": round(median(best_scores), 2) if best_scores else 0.0,
        },
        "reference_metrics": reference_metrics,
        "ranking_stage_metrics": (
            {
                stage_name: _aggregate_rank_stage(query_metrics, stage_name)
                for stage_name in ["raw_retrieval", "reranked", "validation_order", "final_selection"]
            }
            if int(reference_metrics.get("query_count") or 0) > 0
            else {}
        ),
        "metric_protocol": {
            "label_source": "weak_supervised_case_library_anchor",
            "raw_retrieval": "vectorized retrieval score before optional reranking",
            "reranked": "candidate order after optional learned, multimodal, or LLM reranking",
            "validation_order": "deterministic pass gate followed by validation quality score",
            "final_selection": "sample-diverse selection from validation order",
            "legacy_pass_at_k_alias": "final_selection_acceptance_at_k",
            "validation_quality_score": "diagnostic quality score independent of the acceptance gate",
            "best_validation_quality_score": (
                "quality of the policy-selected best case; accepted cases take precedence over rejected cases"
            ),
            "max_validation_quality_score": "maximum quality among selected cases without applying the acceptance gate",
            "validation_score": "legacy alias for validation_quality_score",
        },
        "query_metrics": query_metrics,
        "behavior_coverage": _group_coverage(query_metrics, field_name="behaviors", empty_label="none"),
        "actor_coverage": _group_coverage(query_metrics, field_name="actors", empty_label="any"),
        "tag_coverage": _group_coverage(query_metrics, field_name="tags", empty_label="untagged"),
        "location_distribution": location_distribution,
    }
    return metrics


def _normalize_metric_payload(metrics: Dict[str, object]) -> Dict[str, object]:
    overview = dict(metrics.get("overview") or {})
    overview.setdefault("validation_acceptance_at_1_count", overview.get("pass_at_1_count", 0))
    overview.setdefault("validation_acceptance_at_1_rate", overview.get("pass_at_1_rate", 0.0))
    overview.setdefault("validation_acceptance_at_k_count", overview.get("pass_at_k_count", 0))
    overview.setdefault("validation_acceptance_at_k_rate", overview.get("pass_at_k_rate", 0.0))
    overview.setdefault("final_selection_acceptance_at_1_count", overview.get("pass_at_1_count", 0))
    overview.setdefault("final_selection_acceptance_at_1_rate", overview.get("pass_at_1_rate", 0.0))
    overview.setdefault("final_selection_acceptance_at_k_count", overview.get("pass_at_k_count", 0))
    overview.setdefault("final_selection_acceptance_at_k_rate", overview.get("pass_at_k_rate", 0.0))
    overview.setdefault(
        "mean_best_validation_quality_score",
        overview.get("mean_best_validation_score", 0.0),
    )
    overview.setdefault(
        "median_best_validation_quality_score",
        overview.get("median_best_validation_score", 0.0),
    )
    overview.setdefault(
        "mean_max_validation_quality_score",
        overview.get("mean_best_validation_quality_score", 0.0),
    )
    overview.setdefault(
        "median_max_validation_quality_score",
        overview.get("median_best_validation_quality_score", 0.0),
    )

    rows: List[Dict[str, object]] = []
    for raw_row in list(metrics.get("query_metrics") or []):
        row = dict(raw_row)
        row.setdefault("best_validation_quality_score", get_best_validation_quality_score(row))
        row.setdefault("max_validation_quality_score", get_max_validation_quality_score(row))
        row.setdefault("best_validation_score", row["best_validation_quality_score"])
        final_rank = {
            "reference_objective_at_1": row.get("reference_objective_at_1"),
            "reference_objective_at_k": row.get("reference_objective_at_k"),
        }
        ranking_stages = dict(row.get("ranking_stages") or {})
        for stage_name in ["raw_retrieval", "reranked", "validation_order", "final_selection"]:
            ranking_stages.setdefault(stage_name, dict(final_rank))
        row["ranking_stages"] = ranking_stages
        row.setdefault("validation_acceptance_at_1", bool(row.get("pass_at_1")))
        row.setdefault("validation_acceptance_at_k", bool(row.get("pass_at_k")))
        row.setdefault("final_selection_acceptance_at_1", bool(row.get("pass_at_1")))
        row.setdefault("final_selection_acceptance_at_k", bool(row.get("pass_at_k")))
        rows.append(row)

    behavior_coverage = []
    for raw_row in list(metrics.get("behavior_coverage") or []):
        row = dict(raw_row)
        row.setdefault("avg_best_validation_quality_score", get_average_best_validation_quality_score(row))
        row.setdefault("avg_best_validation_score", row["avg_best_validation_quality_score"])
        behavior_coverage.append(row)

    actor_coverage = []
    for raw_row in list(metrics.get("actor_coverage") or []):
        row = dict(raw_row)
        row.setdefault("avg_best_validation_quality_score", get_average_best_validation_quality_score(row))
        row.setdefault("avg_best_validation_score", row["avg_best_validation_quality_score"])
        actor_coverage.append(row)

    tag_coverage = []
    for raw_row in list(metrics.get("tag_coverage") or []):
        row = dict(raw_row)
        row.setdefault("avg_best_validation_quality_score", get_average_best_validation_quality_score(row))
        row.setdefault("avg_best_validation_score", row["avg_best_validation_quality_score"])
        tag_coverage.append(row)

    reference_metrics = dict(metrics.get("reference_metrics") or {})
    ranking_stage_metrics = dict(metrics.get("ranking_stage_metrics") or {})
    if int(reference_metrics.get("query_count") or 0) <= 0:
        ranking_stage_metrics = {}

    normalized = dict(metrics)
    normalized["overview"] = overview
    normalized["query_metrics"] = rows
    normalized["behavior_coverage"] = behavior_coverage
    normalized["actor_coverage"] = actor_coverage
    normalized["tag_coverage"] = tag_coverage
    normalized["reference_metrics"] = reference_metrics
    normalized["ranking_stage_metrics"] = ranking_stage_metrics
    return normalized


def write_benchmark_metrics(metrics: Dict[str, object], output_dir: Path) -> None:
    metrics = _normalize_metric_payload(metrics)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_metrics.json"
    md_path = output_dir / "benchmark_metrics_summary.md"
    html_path = output_dir / "benchmark_metrics_summary.html"

    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    overview = metrics["overview"]
    lines = [
        "# Benchmark Metrics",
        "",
        "- Queries: {0}".format(overview["query_count"]),
        "- Validation acceptance@1: {0}/{1} ({2:.1%})".format(
            overview["validation_acceptance_at_1_count"],
            overview["query_count"],
            overview["validation_acceptance_at_1_rate"],
        ),
        "- Final selection acceptance@1: {0}/{1} ({2:.1%})".format(
            overview["final_selection_acceptance_at_1_count"],
            overview["query_count"],
            overview["final_selection_acceptance_at_1_rate"],
        ),
        "- Final selection acceptance@K: {0}/{1} ({2:.1%})".format(
            overview["final_selection_acceptance_at_k_count"],
            overview["query_count"],
            overview["final_selection_acceptance_at_k_rate"],
        ),
        "- Selected cases: {0}".format(overview["selected_case_count"]),
        "- Passed selected cases: {0}".format(overview["passed_selected_case_count"]),
        "- Unique cases: {0}".format(overview["unique_case_count"]),
        "- Unique passed cases: {0}".format(overview["unique_passed_case_count"]),
        "- Unique locations: {0}".format(overview["unique_locations"]),
        "- Mean best validation quality: {0:.2f}".format(
            overview.get("mean_best_validation_quality_score", overview["mean_best_validation_score"])
        ),
        "- Mean maximum quality (ungated diagnostic): {0:.2f}".format(
            overview.get("mean_max_validation_quality_score", overview.get("mean_best_validation_quality_score", 0.0))
        ),
        "",
    ]

    ranking_stage_metrics = metrics.get("ranking_stage_metrics") or {}
    if ranking_stage_metrics:
        lines.extend(
            [
                "## Ranking Stages",
                "",
                "| Stage | Reference Objective@1 | Reference Objective@K | Mean Candidates |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for stage_name, stage in ranking_stage_metrics.items():
            lines.append(
                "| `{0}` | {1}/{2} ({3:.1%}) | {4}/{2} ({5:.1%}) | {6:.2f} |".format(
                    stage_name,
                    stage["reference_objective_at_1_count"],
                    stage["query_count"],
                    stage["reference_objective_at_1_rate"],
                    stage["reference_objective_at_k_count"],
                    stage["reference_objective_at_k_rate"],
                    stage["mean_candidate_count"],
                )
            )
        lines.append("")

    reference_metrics = metrics.get("reference_metrics") or {}
    if int(reference_metrics.get("query_count") or 0) > 0:
        lines.extend(
            [
                "## Final-Selection Reference Metrics",
                "",
                "- Labeled queries: {0}".format(reference_metrics["query_count"]),
                "- Scene objective@1: {0}/{1} ({2:.1%})".format(
                    reference_metrics["scene_objective_at_1_count"],
                    reference_metrics["query_count"],
                    reference_metrics["scene_objective_at_1_rate"],
                ),
                "- Scene objective@K: {0}/{1} ({2:.1%})".format(
                    reference_metrics["scene_objective_at_k_count"],
                    reference_metrics["query_count"],
                    reference_metrics["scene_objective_at_k_rate"],
                ),
                "- Actor objective@1: {0}/{1} ({2:.1%})".format(
                    reference_metrics["actor_objective_at_1_count"],
                    reference_metrics["query_count"],
                    reference_metrics["actor_objective_at_1_rate"],
                ),
                "- Actor objective@K: {0}/{1} ({2:.1%})".format(
                    reference_metrics["actor_objective_at_k_count"],
                    reference_metrics["query_count"],
                    reference_metrics["actor_objective_at_k_rate"],
                ),
                "- Reference objective@1: {0}/{1} ({2:.1%})".format(
                    reference_metrics["objective_at_1_count"],
                    reference_metrics["query_count"],
                    reference_metrics["objective_at_1_rate"],
                ),
                "- Reference objective@K: {0}/{1} ({2:.1%})".format(
                    reference_metrics["objective_at_k_count"],
                    reference_metrics["query_count"],
                    reference_metrics["objective_at_k_rate"],
                ),
                "- Positive localization queries: {0}".format(
                    reference_metrics["positive_localization_query_count"]
                ),
                "- Mean event IoU: {0:.3f}".format(reference_metrics["mean_event_iou"]),
                "- Mean peak error: {0:.2f}".format(reference_metrics["mean_peak_error"]),
                "- Contrastive groups: {0}".format(reference_metrics["contrastive_group_count"]),
                "- Contrastive success@1: {0}/{1} ({2:.1%})".format(
                    reference_metrics["contrastive_group_success_at_1_count"],
                    max(1, reference_metrics["contrastive_group_count"]),
                    reference_metrics["contrastive_group_success_at_1_rate"],
                ),
                "- Contrastive success@K: {0}/{1} ({2:.1%})".format(
                    reference_metrics["contrastive_group_success_at_k_count"],
                    max(1, reference_metrics["contrastive_group_count"]),
                    reference_metrics["contrastive_group_success_at_k_rate"],
                ),
                "",
            ]
        )

    lines.extend(
        [
        "## Query Breakdown",
        "",
        "| Query ID | Raw Ref@1 | Reranked Ref@1 | Validation Accept@1 | Final Accept@1 | Final Ref@1 | Final Ref@K | Passed | Selected | Best Validation Quality | Actors | Behaviors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for row in metrics["query_metrics"]:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9:.2f} | {10} | {11} |".format(
                row["id"],
                row["ranking_stages"]["raw_retrieval"].get("reference_objective_at_1")
                if row["ranking_stages"]["raw_retrieval"].get("reference_objective_at_1") is not None
                else "-",
                row["ranking_stages"]["reranked"].get("reference_objective_at_1")
                if row["ranking_stages"]["reranked"].get("reference_objective_at_1") is not None
                else "-",
                row["validation_acceptance_at_1"],
                row["final_selection_acceptance_at_1"],
                row.get("reference_objective_at_1") if row.get("reference_objective_at_1") is not None else "-",
                row.get("reference_objective_at_k") if row.get("reference_objective_at_k") is not None else "-",
                row["passed_count"],
                row["selected_count"],
                row["best_validation_quality_score"],
                ", ".join(row["actors"]) or "any",
                ", ".join(row["behaviors"]) or "none",
            )
        )

    lines.extend(
        [
            "",
            "## Behavior Coverage",
            "",
            "| Behavior | Queries | Final Accept@1 | Final Accept@K | Avg Best Validation Quality |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in metrics["behavior_coverage"]:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4:.2f} |".format(
                row["name"],
                row["query_count"],
                row["pass_at_1_count"],
                row["pass_at_k_count"],
                row["avg_best_validation_quality_score"],
            )
        )

    lines.extend(
        [
            "",
            "## Actor Coverage",
            "",
            "| Actor | Queries | Final Accept@1 | Final Accept@K | Avg Best Validation Quality |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in metrics["actor_coverage"]:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4:.2f} |".format(
                row["name"],
                row["query_count"],
                row["pass_at_1_count"],
                row["pass_at_k_count"],
                row["avg_best_validation_quality_score"],
            )
        )

    lines.extend(
        [
            "",
            "## Location Distribution",
            "",
            "| Location | Cases | Passed |",
            "| --- | --- | --- |",
        ]
    )
    for row in metrics["location_distribution"]:
        lines.append(
            "| {0} | {1} | {2} |".format(
                row["location"],
                row["case_count"],
                row["passed_case_count"],
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_path.write_text(SUMMARY_TEMPLATE.render(metrics=metrics), encoding="utf-8")
