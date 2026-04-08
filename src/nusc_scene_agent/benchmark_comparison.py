from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from jinja2 import Template


DEFAULT_BENCHMARK_PROFILES = [
    {
        "name": "rule_only",
        "label": "Rule-Only",
        "description": "Rule-based parser with deterministic retrieval and validation.",
        "query_mode": "rule",
        "rerank_mode": "none",
    },
    {
        "name": "llm_planner",
        "label": "LLM-Planner",
        "description": "LLM planner with deterministic retrieval and validation.",
        "query_mode": "llm",
        "rerank_mode": "none",
    },
    {
        "name": "hybrid_agent",
        "label": "Hybrid-Agent",
        "description": "Hybrid planner plus LLM reranking and deterministic validation.",
        "query_mode": "hybrid",
        "rerank_mode": "llm",
    },
]


LEADERBOARD_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>nuScenes Scenario Mining Leaderboard</title>
  <style>
    :root {
      --bg: #f3efe8;
      --panel: #fffdfa;
      --ink: #1f1f1f;
      --muted: #6f665c;
      --line: #ddd3c4;
      --head: #e8decf;
      --accent: #264653;
      --good: #2a9d8f;
    }
    body {
      margin: 0;
      padding: 32px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #f8f4ee 0%, var(--bg) 100%);
      color: var(--ink);
    }
    h1, h2 {
      margin: 0 0 12px 0;
    }
    .meta {
      margin: 0 0 20px 0;
      color: var(--muted);
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px 22px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(31, 31, 31, 0.05);
    }
    .pill {
      display: inline-block;
      margin-right: 10px;
      margin-bottom: 10px;
      padding: 8px 12px;
      border-radius: 999px;
      background: #f1ebe2;
      color: var(--accent);
      font-size: 13px;
      font-weight: 600;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      background: white;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      background: var(--head);
      font-weight: 700;
    }
    .rank {
      font-weight: 700;
      color: var(--accent);
    }
    .top {
      color: var(--good);
      font-weight: 700;
    }
    .small {
      color: var(--muted);
      font-size: 13px;
    }
    code {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.95em;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>nuScenes Scenario Mining Leaderboard</h1>
    <p class="meta">
      Profiles: {{ comparison.overview.profile_count }} |
      Queries: {{ comparison.overview.query_count }} |
      Planner disagreement count: {{ comparison.overview.signal_divergence_count }}
    </p>
    <div>
      <span class="pill">Sort: Scenario Group Success@1</span>
      <span class="pill">Tie-break: Scene@1, Actor@1, Mean Event IoU, Mean Best Score</span>
    </div>
  </div>

  <div class="card">
    <h2>Leaderboard</h2>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Profile</th>
          <th>Mode</th>
          <th>Pass@1</th>
          <th>Scene@1</th>
          <th>Actor@1</th>
          <th>Reference@1</th>
          <th>Scenario Group Success@1</th>
          <th>Mean Event IoU</th>
          <th>Mean Peak Error</th>
          <th>Mean Best Score</th>
        </tr>
      </thead>
      <tbody>
      {% for row in comparison.leaderboard %}
        <tr>
          <td class="rank">{{ row.rank }}</td>
          <td class="{{ 'top' if row.rank == 1 else '' }}">{{ row.label }}</td>
          <td><code>{{ row.query_mode }}</code> + <code>{{ row.rerank_mode }}</code></td>
          <td>{{ row.pass_at_1_count }}/{{ row.query_count }} ({{ "%.1f"|format(row.pass_at_1_rate * 100.0) }}%)</td>
          <td>{{ row.scene_objective_at_1_count }}/{{ row.reference_query_count }} ({{ "%.1f"|format(row.scene_objective_at_1_rate * 100.0) }}%)</td>
          <td>{{ row.actor_objective_at_1_count }}/{{ row.reference_query_count }} ({{ "%.1f"|format(row.actor_objective_at_1_rate * 100.0) }}%)</td>
          <td>{{ row.reference_objective_at_1_count }}/{{ row.reference_query_count }} ({{ "%.1f"|format(row.reference_objective_at_1_rate * 100.0) }}%)</td>
          <td>{{ row.scenario_group_scene_success_at_1_count }}/{{ row.scenario_group_count }} ({{ "%.1f"|format(row.scenario_group_scene_success_at_1_rate * 100.0) }}%)</td>
          <td>{{ "%.3f"|format(row.mean_event_iou) }}</td>
          <td>{{ "%.2f"|format(row.mean_peak_error) }}</td>
          <td>{{ "%.2f"|format(row.mean_best_validation_score) }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Behavior Decomposition</h2>
    <table>
      <thead>
        <tr>
          <th>Behavior</th>
          <th>Queries</th>
          <th>Divergent Queries</th>
          <th>Best-Profile Wins</th>
          <th>Scene@1 Summary</th>
        </tr>
      </thead>
      <tbody>
      {% for row in comparison.behavior_error_analysis %}
        <tr>
          <td><code>{{ row.behavior }}</code></td>
          <td>{{ row.query_count }}</td>
          <td>{{ row.divergent_query_count }}</td>
          <td>{{ row.best_profile_summary }}</td>
          <td>{{ row.scene_at_1_summary }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    <p class="small">Use the CSV and markdown exports for full per-profile failure-mode details.</p>
  </div>
</body>
</html>
"""
)


BEHAVIOR_ANALYSIS_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>nuScenes Behavior Error Analysis</title>
  <style>
    body {
      margin: 0;
      padding: 32px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: #f5f1ea;
      color: #1f1f1f;
    }
    h1, h2, h3 {
      margin-bottom: 12px;
    }
    .card {
      background: #fffdfa;
      border: 1px solid #ddd3c4;
      border-radius: 16px;
      padding: 20px 22px;
      margin-bottom: 20px;
    }
    .meta {
      color: #6f665c;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      background: white;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid #ddd3c4;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #e8decf;
    }
    code {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Behavior Error Analysis</h1>
    <p class="meta">
      Behavior groups: {{ analysis|length }} |
      Profiles: {{ profile_labels.values()|join(", ") }}
    </p>
  </div>

  {% for row in analysis %}
  <div class="card">
    <h2><code>{{ row.behavior }}</code></h2>
    <p class="meta">
      Queries: {{ row.query_count }} |
      Divergent queries: {{ row.divergent_query_count }} |
      Best-profile wins: {{ row.best_profile_summary }}
    </p>
    <table>
      <thead>
        <tr>
          <th>Profile</th>
          <th>Pass@1</th>
          <th>Scene@1</th>
          <th>Actor@1</th>
          <th>Reference@1</th>
          <th>Mean Event IoU</th>
          <th>Mean Peak Error</th>
          <th>Top Failure Modes</th>
        </tr>
      </thead>
      <tbody>
      {% for name in profile_order %}
        {% set item = row.profile_summary[name] %}
        <tr>
          <td>{{ item.label }}</td>
          <td>{{ item.pass_at_1_count }}/{{ item.query_count }} ({{ "%.1f"|format(item.pass_at_1_rate * 100.0) }}%)</td>
          <td>{{ item.scene_objective_at_1_count }}/{{ item.query_count }} ({{ "%.1f"|format(item.scene_objective_at_1_rate * 100.0) }}%)</td>
          <td>{{ item.actor_objective_at_1_count }}/{{ item.query_count }} ({{ "%.1f"|format(item.actor_objective_at_1_rate * 100.0) }}%)</td>
          <td>{{ item.reference_objective_at_1_count }}/{{ item.query_count }} ({{ "%.1f"|format(item.reference_objective_at_1_rate * 100.0) }}%)</td>
          <td>{{ "%.3f"|format(item.mean_event_iou) }}</td>
          <td>{{ "%.2f"|format(item.mean_peak_error) }}</td>
          <td>{{ item.top_failure_summary }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% endfor %}
</body>
</html>
"""
)


def default_benchmark_profiles(include_llm: bool = True) -> List[Dict[str, str]]:
    if include_llm:
        return [dict(item) for item in DEFAULT_BENCHMARK_PROFILES]
    return [dict(DEFAULT_BENCHMARK_PROFILES[0])]


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return _load_json(path)


def _top_taxonomy_label(taxonomy: Dict[str, object]) -> str:
    label_distribution = list(taxonomy.get("label_distribution") or [])
    if not label_distribution:
        return "none"
    return str(label_distribution[0].get("name") or "none")


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _sorted_unique(items: Sequence[str]) -> List[str]:
    return sorted({str(item) for item in items if item})


def _same_set(lhs: Sequence[str], rhs: Sequence[str]) -> bool:
    return _sorted_unique(lhs) == _sorted_unique(rhs)


def _top_failure_modes(failure_counts: Dict[str, int], limit: int = 3) -> List[Dict[str, object]]:
    rows = [{"name": name, "count": int(count)} for name, count in failure_counts.items() if int(count) > 0]
    rows.sort(key=lambda item: (int(item["count"]), str(item["name"])), reverse=True)
    return rows[:limit]


def _summary_failure_text(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return "none"
    return ", ".join("{0}:{1}".format(item["name"], item["count"]) for item in rows)


def _build_leaderboard(profile_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    leaderboard_rows: List[Dict[str, object]] = []
    for row in profile_rows:
        scenario_group_count = int(row.get("scenario_group_count") or 0)
        leaderboard_rows.append(
            {
                **dict(row),
                "scenario_group_scene_success_at_1_rate": _ratio(
                    int(row.get("scenario_group_scene_success_at_1_count") or 0),
                    scenario_group_count,
                ),
                "scenario_group_actor_success_at_1_rate": _ratio(
                    int(row.get("scenario_group_actor_success_at_1_count") or 0),
                    scenario_group_count,
                ),
                "scenario_group_reference_success_at_1_rate": _ratio(
                    int(row.get("scenario_group_reference_success_at_1_count") or 0),
                    scenario_group_count,
                ),
                "scene_objective_at_1_count": int(
                    round(float(row.get("scene_objective_at_1_rate") or 0.0) * int(row.get("reference_query_count") or 0))
                ),
                "actor_objective_at_1_count": int(
                    round(float(row.get("actor_objective_at_1_rate") or 0.0) * int(row.get("reference_query_count") or 0))
                ),
                "reference_objective_at_1_count": int(
                    round(float(row.get("reference_objective_at_1_rate") or 0.0) * int(row.get("reference_query_count") or 0))
                ),
            }
        )

    leaderboard_rows.sort(
        key=lambda item: (
            float(item["scenario_group_scene_success_at_1_rate"]),
            float(item["scene_objective_at_1_rate"]),
            float(item["actor_objective_at_1_rate"]),
            float(item["reference_objective_at_1_rate"]),
            float(item["mean_event_iou"]),
            float(item["mean_best_validation_score"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(leaderboard_rows, start=1):
        row["rank"] = rank
    return leaderboard_rows


def _build_behavior_error_analysis(
    query_rows: Sequence[Dict[str, object]],
    profile_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    profile_order = [str(profile["name"]) for profile in profile_rows]
    profile_labels = {str(profile["name"]): str(profile["label"]) for profile in profile_rows}
    grouped: Dict[str, Dict[str, object]] = {}

    for row in query_rows:
        behaviors = [str(item) for item in list(row.get("behaviors") or [])] or ["none"]
        expected_behaviors = list(row.get("behaviors") or [])
        expected_actors = list(row.get("actors") or [])
        for behavior in behaviors:
            bucket = grouped.setdefault(
                behavior,
                {
                    "behavior": behavior,
                    "query_ids": [],
                    "divergent_query_ids": [],
                    "best_profile_counts": defaultdict(int),
                    "profiles": {
                        name: {
                            "query_count": 0,
                            "pass_at_1_count": 0,
                            "scene_objective_at_1_count": 0,
                            "actor_objective_at_1_count": 0,
                            "reference_objective_at_1_count": 0,
                            "event_ious": [],
                            "peak_errors": [],
                            "failure_mode_counts": defaultdict(int),
                            "failure_query_ids": set(),
                        }
                        for name in profile_order
                    },
                },
            )
            bucket["query_ids"].append(str(row["id"]))
            if row.get("signal_divergence"):
                bucket["divergent_query_ids"].append(str(row["id"]))
            bucket["best_profile_counts"][str(row["best_profile"])] += 1

            for profile_name in profile_order:
                metrics = dict(row["profiles"].get(profile_name) or {})
                profile_bucket = dict(bucket["profiles"][profile_name])
                profile_bucket["query_count"] += 1
                profile_bucket["pass_at_1_count"] += int(bool(metrics.get("pass_at_1")))
                profile_bucket["scene_objective_at_1_count"] += int(bool(metrics.get("scene_objective_at_1")))
                profile_bucket["actor_objective_at_1_count"] += int(bool(metrics.get("actor_objective_at_1")))
                profile_bucket["reference_objective_at_1_count"] += int(bool(metrics.get("reference_objective_at_1")))

                if metrics.get("event_iou") is not None:
                    profile_bucket["event_ious"].append(float(metrics["event_iou"]))
                if metrics.get("peak_error") is not None:
                    profile_bucket["peak_errors"].append(float(metrics["peak_error"]))

                failure_mode_counts = profile_bucket["failure_mode_counts"]
                failure_query_ids = profile_bucket["failure_query_ids"]
                query_id = str(row["id"])
                if metrics.get("scene_objective_at_1") is False:
                    failure_mode_counts["scene_mismatch"] += 1
                    failure_query_ids.add(query_id)
                if metrics.get("actor_objective_at_1") is False:
                    failure_mode_counts["actor_mismatch"] += 1
                    failure_query_ids.add(query_id)
                if metrics.get("reference_objective_at_1") is False:
                    failure_mode_counts["anchor_mismatch"] += 1
                    failure_query_ids.add(query_id)
                if metrics.get("event_iou") is not None and float(metrics["event_iou"]) < 0.5:
                    failure_mode_counts["event_window_drift"] += 1
                    failure_query_ids.add(query_id)
                if metrics.get("peak_error") is not None and float(metrics["peak_error"]) > 2.0:
                    failure_mode_counts["peak_shift"] += 1
                    failure_query_ids.add(query_id)
                if not _same_set(expected_behaviors, metrics.get("resolved_behaviors") or []):
                    failure_mode_counts["behavior_parse_shift"] += 1
                    failure_query_ids.add(query_id)
                if expected_actors and not _same_set(expected_actors, metrics.get("resolved_category_groups") or []):
                    failure_mode_counts["actor_parse_shift"] += 1
                    failure_query_ids.add(query_id)

                bucket["profiles"][profile_name] = profile_bucket

    rows: List[Dict[str, object]] = []
    for behavior, bucket in grouped.items():
        query_ids = _sorted_unique(bucket["query_ids"])
        divergent_query_ids = _sorted_unique(bucket["divergent_query_ids"])
        best_profile_counts = {name: int(count) for name, count in dict(bucket["best_profile_counts"]).items()}
        best_profile_summary = ", ".join(
            "{0}:{1}".format(profile_labels.get(name, name), best_profile_counts.get(name, 0))
            for name in profile_order
        )

        profile_summary: Dict[str, Dict[str, object]] = {}
        profile_rows_for_behavior: List[Dict[str, object]] = []
        for profile_name in profile_order:
            profile_bucket = dict(bucket["profiles"][profile_name])
            top_failure_modes = _top_failure_modes(profile_bucket["failure_mode_counts"])
            summary = {
                "name": profile_name,
                "label": profile_labels.get(profile_name, profile_name),
                "query_count": int(profile_bucket["query_count"]),
                "best_profile_query_wins": int(best_profile_counts.get(profile_name, 0)),
                "pass_at_1_count": int(profile_bucket["pass_at_1_count"]),
                "pass_at_1_rate": _ratio(int(profile_bucket["pass_at_1_count"]), int(profile_bucket["query_count"])),
                "scene_objective_at_1_count": int(profile_bucket["scene_objective_at_1_count"]),
                "scene_objective_at_1_rate": _ratio(
                    int(profile_bucket["scene_objective_at_1_count"]),
                    int(profile_bucket["query_count"]),
                ),
                "actor_objective_at_1_count": int(profile_bucket["actor_objective_at_1_count"]),
                "actor_objective_at_1_rate": _ratio(
                    int(profile_bucket["actor_objective_at_1_count"]),
                    int(profile_bucket["query_count"]),
                ),
                "reference_objective_at_1_count": int(profile_bucket["reference_objective_at_1_count"]),
                "reference_objective_at_1_rate": _ratio(
                    int(profile_bucket["reference_objective_at_1_count"]),
                    int(profile_bucket["query_count"]),
                ),
                "mean_event_iou": round(mean(profile_bucket["event_ious"]), 4) if profile_bucket["event_ious"] else 0.0,
                "mean_peak_error": round(mean(profile_bucket["peak_errors"]), 2) if profile_bucket["peak_errors"] else 0.0,
                "failure_query_count": len(profile_bucket["failure_query_ids"]),
                "failure_query_ids": sorted(str(item) for item in profile_bucket["failure_query_ids"]),
                "top_failure_modes": top_failure_modes,
                "top_failure_summary": _summary_failure_text(top_failure_modes),
            }
            profile_summary[profile_name] = summary
            profile_rows_for_behavior.append(summary)

        rows.append(
            {
                "behavior": behavior,
                "query_count": len(query_ids),
                "query_ids": query_ids,
                "divergent_query_count": len(divergent_query_ids),
                "divergent_query_ids": divergent_query_ids,
                "best_profile_counts": best_profile_counts,
                "best_profile_summary": best_profile_summary,
                "scene_at_1_summary": ", ".join(
                    "{0}:{1}/{2}".format(
                        profile_summary[name]["label"],
                        profile_summary[name]["scene_objective_at_1_count"],
                        profile_summary[name]["query_count"],
                    )
                    for name in profile_order
                ),
                "profile_summary": profile_summary,
                "profiles": profile_rows_for_behavior,
            }
        )

    rows.sort(key=lambda item: (int(item["query_count"]), int(item["divergent_query_count"]), str(item["behavior"])), reverse=True)
    return rows


def build_benchmark_comparison(profile_runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    profiles: List[Dict[str, object]] = []
    query_table: Dict[str, Dict[str, object]] = {}

    for run in profile_runs:
        output_dir = Path(run["output_dir"]).resolve()
        metrics = _load_json(output_dir / "benchmark_metrics.json")
        taxonomy = _load_json(output_dir / "hard_case_taxonomy.json")
        scenario_summary = _load_optional_json(output_dir / "scenario_group_summary.json")

        overview = dict(metrics.get("overview") or {})
        reference_metrics = dict(metrics.get("reference_metrics") or {})
        scenario_overview = dict(scenario_summary.get("overview") or {})
        profile_entry = {
            "name": str(run["name"]),
            "label": str(run.get("label") or run["name"]),
            "description": str(run.get("description") or ""),
            "query_mode": str(run.get("query_mode") or ""),
            "rerank_mode": str(run.get("rerank_mode") or ""),
            "output_dir": str(output_dir),
            "query_count": int(overview.get("query_count") or 0),
            "pass_at_1_count": int(overview.get("pass_at_1_count") or 0),
            "pass_at_1_rate": float(overview.get("pass_at_1_rate") or 0.0),
            "pass_at_k_count": int(overview.get("pass_at_k_count") or 0),
            "pass_at_k_rate": float(overview.get("pass_at_k_rate") or 0.0),
            "mean_best_validation_score": float(overview.get("mean_best_validation_score") or 0.0),
            "unique_case_count": int(overview.get("unique_case_count") or 0),
            "hard_case_count": int((taxonomy.get("overview") or {}).get("hard_case_count") or 0),
            "failed_hard_case_count": int((taxonomy.get("overview") or {}).get("failed_count") or 0),
            "top_taxonomy_label": _top_taxonomy_label(taxonomy),
            "reference_query_count": int(reference_metrics.get("query_count") or 0),
            "scene_objective_at_1_rate": float(reference_metrics.get("scene_objective_at_1_rate") or 0.0),
            "scene_objective_at_k_rate": float(reference_metrics.get("scene_objective_at_k_rate") or 0.0),
            "actor_objective_at_1_rate": float(reference_metrics.get("actor_objective_at_1_rate") or 0.0),
            "actor_objective_at_k_rate": float(reference_metrics.get("actor_objective_at_k_rate") or 0.0),
            "reference_objective_at_1_rate": float(reference_metrics.get("objective_at_1_rate") or 0.0),
            "reference_objective_at_k_rate": float(reference_metrics.get("objective_at_k_rate") or 0.0),
            "mean_event_iou": float(reference_metrics.get("mean_event_iou") or 0.0),
            "mean_peak_error": float(reference_metrics.get("mean_peak_error") or 0.0),
            "scenario_group_count": int(scenario_overview.get("group_count") or 0),
            "scenario_group_scene_success_at_1_count": int(scenario_overview.get("scene_success_at_1_count") or 0),
            "scenario_group_scene_success_at_k_count": int(scenario_overview.get("scene_success_at_k_count") or 0),
            "scenario_group_actor_success_at_1_count": int(scenario_overview.get("actor_success_at_1_count") or 0),
            "scenario_group_actor_success_at_k_count": int(scenario_overview.get("actor_success_at_k_count") or 0),
            "scenario_group_reference_success_at_1_count": int(scenario_overview.get("reference_success_at_1_count") or 0),
            "scenario_group_reference_success_at_k_count": int(scenario_overview.get("reference_success_at_k_count") or 0),
            "scenario_group_mean_event_iou": float(scenario_overview.get("mean_event_iou") or 0.0),
            "scenario_group_mean_peak_error": float(scenario_overview.get("mean_peak_error") or 0.0),
        }
        profiles.append(profile_entry)

        for row in metrics.get("query_metrics") or []:
            query_id = str(row.get("id") or "")
            query_entry = query_table.setdefault(
                query_id,
                {
                    "id": query_id,
                    "description": str(row.get("description") or ""),
                    "actors": list(row.get("actors") or []),
                    "behaviors": list(row.get("behaviors") or []),
                    "profiles": {},
                },
            )
            query_entry["profiles"][profile_entry["name"]] = {
                "pass_at_1": bool(row.get("pass_at_1")),
                "pass_at_k": bool(row.get("pass_at_k")),
                "best_validation_score": float(row.get("best_validation_score") or 0.0),
                "selected_count": int(row.get("selected_count") or 0),
                "passed_count": int(row.get("passed_count") or 0),
                "scene_objective_at_1": row.get("scene_objective_at_1"),
                "scene_objective_at_k": row.get("scene_objective_at_k"),
                "actor_objective_at_1": row.get("actor_objective_at_1"),
                "actor_objective_at_k": row.get("actor_objective_at_k"),
                "reference_objective_at_1": row.get("reference_objective_at_1"),
                "reference_objective_at_k": row.get("reference_objective_at_k"),
                "event_iou": row.get("event_iou"),
                "peak_error": row.get("peak_error"),
                "resolved_category_groups": list(row.get("resolved_category_groups") or []),
                "resolved_positions": list(row.get("resolved_positions") or []),
                "resolved_behaviors": list(row.get("resolved_behaviors") or []),
                "resolved_risk_terms": list(row.get("resolved_risk_terms") or []),
            }

    profile_name_order = [str(profile["name"]) for profile in profiles]
    query_comparison: List[Dict[str, object]] = []
    for query_id, row in sorted(query_table.items()):
        best_profile = ""
        best_score = -1.0
        signatures = set()
        for profile_name in profile_name_order:
            profile_metrics = dict(row["profiles"].get(profile_name) or {})
            score = float(profile_metrics.get("best_validation_score") or 0.0)
            if score > best_score:
                best_profile = profile_name
                best_score = score
            signatures.add(
                (
                    tuple(profile_metrics.get("resolved_category_groups") or []),
                    tuple(profile_metrics.get("resolved_positions") or []),
                    tuple(profile_metrics.get("resolved_behaviors") or []),
                    tuple(profile_metrics.get("resolved_risk_terms") or []),
                )
            )

        query_comparison.append(
            {
                "id": query_id,
                "description": row["description"],
                "actors": row["actors"],
                "behaviors": row["behaviors"],
                "profiles": row["profiles"],
                "best_profile": best_profile,
                "signal_divergence": len(signatures) > 1,
                "score_span": round(
                    max(
                        float((row["profiles"].get(profile_name) or {}).get("best_validation_score") or 0.0)
                        for profile_name in profile_name_order
                    )
                    - min(
                        float((row["profiles"].get(profile_name) or {}).get("best_validation_score") or 0.0)
                        for profile_name in profile_name_order
                    ),
                    2,
                ),
            }
        )

    deltas_vs_baseline: List[Dict[str, object]] = []
    if profiles:
        baseline_name = str(profiles[0]["name"])
        baseline_rows = {row["id"]: dict(row["profiles"].get(baseline_name) or {}) for row in query_comparison}

        for profile in profiles[1:]:
            profile_name = str(profile["name"])
            improved_queries = 0
            unchanged_queries = 0
            degraded_queries = 0
            pass_at_1_gain = 0
            pass_at_k_gain = 0
            score_deltas: List[float] = []

            for row in query_comparison:
                baseline = baseline_rows.get(row["id"], {})
                current = dict(row["profiles"].get(profile_name) or {})
                baseline_score = float(baseline.get("best_validation_score") or 0.0)
                current_score = float(current.get("best_validation_score") or 0.0)
                score_delta = round(current_score - baseline_score, 2)
                score_deltas.append(score_delta)

                if score_delta > 0.01:
                    improved_queries += 1
                elif score_delta < -0.01:
                    degraded_queries += 1
                else:
                    unchanged_queries += 1

                pass_at_1_gain += int(bool(current.get("pass_at_1"))) - int(bool(baseline.get("pass_at_1")))
                pass_at_k_gain += int(bool(current.get("pass_at_k"))) - int(bool(baseline.get("pass_at_k")))

            deltas_vs_baseline.append(
                {
                    "baseline": baseline_name,
                    "profile": profile_name,
                    "improved_queries": improved_queries,
                    "unchanged_queries": unchanged_queries,
                    "degraded_queries": degraded_queries,
                    "mean_score_delta": round(mean(score_deltas), 2) if score_deltas else 0.0,
                    "pass_at_1_gain": pass_at_1_gain,
                    "pass_at_k_gain": pass_at_k_gain,
                }
            )

    leaderboard = _build_leaderboard(profiles)
    behavior_error_analysis = _build_behavior_error_analysis(query_comparison, profiles)
    return {
        "overview": {
            "profile_count": len(profiles),
            "query_count": len(query_comparison),
            "profile_names": profile_name_order,
            "signal_divergence_count": sum(1 for row in query_comparison if row["signal_divergence"]),
        },
        "profiles": profiles,
        "leaderboard": leaderboard,
        "behavior_error_analysis": behavior_error_analysis,
        "deltas_vs_baseline": deltas_vs_baseline,
        "query_comparison": query_comparison,
    }


def write_benchmark_comparison(comparison: Dict[str, object], output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_profile_comparison.json"
    md_path = output_dir / "benchmark_profile_comparison_summary.md"
    leaderboard_csv_path = output_dir / "benchmark_leaderboard.csv"
    leaderboard_html_path = output_dir / "benchmark_leaderboard.html"
    behavior_json_path = output_dir / "behavior_error_analysis.json"
    behavior_md_path = output_dir / "behavior_error_analysis.md"
    behavior_csv_path = output_dir / "behavior_error_analysis.csv"
    behavior_html_path = output_dir / "behavior_error_analysis.html"

    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    profile_rows = list(comparison.get("profiles") or [])
    query_rows = list(comparison.get("query_comparison") or [])
    leaderboard_rows = list(comparison.get("leaderboard") or [])
    behavior_rows = list(comparison.get("behavior_error_analysis") or [])

    lines = [
        "# Benchmark Profile Comparison",
        "",
        "- Profiles: {0}".format(comparison["overview"]["profile_count"]),
        "- Queries: {0}".format(comparison["overview"]["query_count"]),
        "- Queries with planner disagreement: {0}".format(comparison["overview"]["signal_divergence_count"]),
        "",
        "## Profile Overview",
        "",
        "| Profile | Query Mode | Rerank | Pass@1 | Pass@K | Mean Best Score | Unique Cases | Hard Cases | Top Taxonomy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in profile_rows:
        lines.append(
            "| {0} | {1} | {2} | {3}/{4} ({5:.1%}) | {6}/{4} ({7:.1%}) | {8:.2f} | {9} | {10} | {11} |".format(
                row["label"],
                row["query_mode"],
                row["rerank_mode"],
                row["pass_at_1_count"],
                row["query_count"],
                row["pass_at_1_rate"],
                row["pass_at_k_count"],
                row["pass_at_k_rate"],
                row["mean_best_validation_score"],
                row["unique_case_count"],
                row["hard_case_count"],
                row["top_taxonomy_label"],
            )
        )

    if leaderboard_rows:
        lines.extend(
            [
                "",
                "## Leaderboard",
                "",
                "| Rank | Profile | Pass@1 | Scene@1 | Actor@1 | Reference@1 | Scenario Group Success@1 | Mean Event IoU | Mean Peak Error |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in leaderboard_rows:
            lines.append(
                "| {0} | {1} | {2}/{3} ({4:.1%}) | {5}/{6} ({7:.1%}) | {8}/{6} ({9:.1%}) | {10}/{6} ({11:.1%}) | {12}/{13} ({14:.1%}) | {15:.3f} | {16:.2f} |".format(
                    row["rank"],
                    row["label"],
                    row["pass_at_1_count"],
                    row["query_count"],
                    row["pass_at_1_rate"],
                    row["scene_objective_at_1_count"],
                    row["reference_query_count"],
                    row["scene_objective_at_1_rate"],
                    row["actor_objective_at_1_count"],
                    row["actor_objective_at_1_rate"],
                    row["reference_objective_at_1_count"],
                    row["reference_objective_at_1_rate"],
                    row["scenario_group_scene_success_at_1_count"],
                    row["scenario_group_count"],
                    row["scenario_group_scene_success_at_1_rate"],
                    row["mean_event_iou"],
                    row["mean_peak_error"],
                )
            )

    if comparison.get("deltas_vs_baseline"):
        lines.extend(
            [
                "",
                "## Delta vs Baseline",
                "",
                "| Profile | Improved Queries | Unchanged | Degraded | Mean Score Delta | Pass@1 Gain | Pass@K Gain |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in comparison["deltas_vs_baseline"]:
            lines.append(
                "| {0} | {1} | {2} | {3} | {4:.2f} | {5} | {6} |".format(
                    row["profile"],
                    row["improved_queries"],
                    row["unchanged_queries"],
                    row["degraded_queries"],
                    row["mean_score_delta"],
                    row["pass_at_1_gain"],
                    row["pass_at_k_gain"],
                )
            )

    reference_profiles = [row for row in profile_rows if int(row.get("reference_query_count") or 0) > 0]
    if reference_profiles:
        lines.extend(
            [
                "",
                "## Reference-Aware Query Metrics",
                "",
                "| Profile | Labeled Queries | Scene@1 | Scene@K | Actor@1 | Actor@K | Reference@1 | Reference@K | Mean Event IoU | Mean Peak Error |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in reference_profiles:
            lines.append(
                "| {0} | {1} | {2:.1%} | {3:.1%} | {4:.1%} | {5:.1%} | {6:.1%} | {7:.1%} | {8:.3f} | {9:.2f} |".format(
                    row["label"],
                    row["reference_query_count"],
                    row["scene_objective_at_1_rate"],
                    row["scene_objective_at_k_rate"],
                    row["actor_objective_at_1_rate"],
                    row["actor_objective_at_k_rate"],
                    row["reference_objective_at_1_rate"],
                    row["reference_objective_at_k_rate"],
                    row["mean_event_iou"],
                    row["mean_peak_error"],
                )
            )

    scenario_profiles = [row for row in profile_rows if int(row.get("scenario_group_count") or 0) > 0]
    if scenario_profiles:
        lines.extend(
            [
                "",
                "## Scenario Group Metrics",
                "",
                "| Profile | Groups | Scene@1 | Scene@K | Actor@1 | Actor@K | Reference@1 | Reference@K | Mean Event IoU | Mean Peak Error |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in scenario_profiles:
            group_count = int(row["scenario_group_count"])
            lines.append(
                "| {0} | {1} | {2}/{1} ({3:.1%}) | {4}/{1} ({5:.1%}) | {6}/{1} ({7:.1%}) | {8}/{1} ({9:.1%}) | {10}/{1} ({11:.1%}) | {12}/{1} ({13:.1%}) | {14:.3f} | {15:.2f} |".format(
                    row["label"],
                    group_count,
                    row["scenario_group_scene_success_at_1_count"],
                    row["scenario_group_scene_success_at_1_count"] / group_count if group_count else 0.0,
                    row["scenario_group_scene_success_at_k_count"],
                    row["scenario_group_scene_success_at_k_count"] / group_count if group_count else 0.0,
                    row["scenario_group_actor_success_at_1_count"],
                    row["scenario_group_actor_success_at_1_count"] / group_count if group_count else 0.0,
                    row["scenario_group_actor_success_at_k_count"],
                    row["scenario_group_actor_success_at_k_count"] / group_count if group_count else 0.0,
                    row["scenario_group_reference_success_at_1_count"],
                    row["scenario_group_reference_success_at_1_count"] / group_count if group_count else 0.0,
                    row["scenario_group_reference_success_at_k_count"],
                    row["scenario_group_reference_success_at_k_count"] / group_count if group_count else 0.0,
                    row["scenario_group_mean_event_iou"],
                    row["scenario_group_mean_peak_error"],
                )
            )

    if behavior_rows:
        lines.extend(
            [
                "",
                "## Behavior Breakdown",
                "",
                "| Behavior | Queries | Divergent Queries | Best-Profile Wins |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in behavior_rows:
            lines.append(
                "| {0} | {1} | {2} | {3} |".format(
                    row["behavior"],
                    row["query_count"],
                    row["divergent_query_count"],
                    row["best_profile_summary"],
                )
            )

    profile_names = [str(profile["name"]) for profile in profile_rows]
    profile_labels = {str(profile["name"]): str(profile["label"]) for profile in profile_rows}
    if query_rows:
        header = ["Query ID", "Actors", "Behaviors"] + [profile_labels[name] for name in profile_names] + ["Best Profile"]
        lines.extend(["", "## Query Matrix", "", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"])
        for row in query_rows:
            cells = [
                row["id"],
                ", ".join(row["actors"]) or "any",
                ", ".join(row["behaviors"]) or "none",
            ]
            for profile_name in profile_names:
                metrics = dict(row["profiles"].get(profile_name) or {})
                cells.append(
                    "{0}/{1:.2f}".format(
                        "T" if metrics.get("pass_at_1") else "F",
                        float(metrics.get("best_validation_score") or 0.0),
                    )
                )
            cells.append(profile_labels.get(str(row["best_profile"]), str(row["best_profile"])))
            lines.append("| " + " | ".join(cells) + " |")

    divergent_rows = [row for row in query_rows if row.get("signal_divergence")]
    if divergent_rows:
        lines.extend(
            [
                "",
                "## Planner Disagreement",
                "",
                "| Query ID | Profile | Actors | Positions | Behaviors | Risk Terms |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in divergent_rows:
            for profile_name in profile_names:
                metrics = dict(row["profiles"].get(profile_name) or {})
                lines.append(
                    "| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                        row["id"],
                        profile_labels.get(profile_name, profile_name),
                        ", ".join(metrics.get("resolved_category_groups") or []) or "none",
                        ", ".join(metrics.get("resolved_positions") or []) or "none",
                        ", ".join(metrics.get("resolved_behaviors") or []) or "none",
                        ", ".join(metrics.get("resolved_risk_terms") or []) or "none",
                    )
                )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with leaderboard_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "name",
                "label",
                "query_mode",
                "rerank_mode",
                "query_count",
                "pass_at_1_count",
                "pass_at_1_rate",
                "scene_objective_at_1_count",
                "reference_query_count",
                "scene_objective_at_1_rate",
                "actor_objective_at_1_count",
                "actor_objective_at_1_rate",
                "reference_objective_at_1_count",
                "reference_objective_at_1_rate",
                "scenario_group_count",
                "scenario_group_scene_success_at_1_count",
                "scenario_group_scene_success_at_1_rate",
                "mean_event_iou",
                "mean_peak_error",
                "mean_best_validation_score",
                "hard_case_count",
                "top_taxonomy_label",
            ],
        )
        writer.writeheader()
        for row in leaderboard_rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})

    behavior_json_path.write_text(json.dumps(behavior_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    behavior_lines = [
        "# Behavior Error Analysis",
        "",
        "- Behavior groups: {0}".format(len(behavior_rows)),
        "- Profiles: {0}".format(", ".join(profile_labels[name] for name in profile_names)),
        "",
    ]
    for row in behavior_rows:
        behavior_lines.extend(
            [
                "## {0}".format(row["behavior"]),
                "",
                "- Queries: {0}".format(row["query_count"]),
                "- Divergent queries: {0}".format(row["divergent_query_count"]),
                "- Best-profile wins: {0}".format(row["best_profile_summary"]),
                "",
                "| Profile | Pass@1 | Scene@1 | Actor@1 | Reference@1 | Mean Event IoU | Mean Peak Error | Top Failure Modes |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name in profile_names:
            item = row["profile_summary"][name]
            behavior_lines.append(
                "| {0} | {1}/{2} ({3:.1%}) | {4}/{2} ({5:.1%}) | {6}/{2} ({7:.1%}) | {8}/{2} ({9:.1%}) | {10:.3f} | {11:.2f} | {12} |".format(
                    item["label"],
                    item["pass_at_1_count"],
                    item["query_count"],
                    item["pass_at_1_rate"],
                    item["scene_objective_at_1_count"],
                    item["scene_objective_at_1_rate"],
                    item["actor_objective_at_1_count"],
                    item["actor_objective_at_1_rate"],
                    item["reference_objective_at_1_count"],
                    item["reference_objective_at_1_rate"],
                    item["mean_event_iou"],
                    item["mean_peak_error"],
                    item["top_failure_summary"],
                )
            )
        behavior_lines.append("")
    behavior_md_path.write_text("\n".join(behavior_lines), encoding="utf-8")

    with behavior_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "behavior",
                "profile",
                "profile_label",
                "query_count",
                "divergent_query_count",
                "best_profile_query_wins",
                "pass_at_1_count",
                "pass_at_1_rate",
                "scene_objective_at_1_count",
                "scene_objective_at_1_rate",
                "actor_objective_at_1_count",
                "actor_objective_at_1_rate",
                "reference_objective_at_1_count",
                "reference_objective_at_1_rate",
                "mean_event_iou",
                "mean_peak_error",
                "failure_query_count",
                "top_failure_1",
                "top_failure_1_count",
                "top_failure_2",
                "top_failure_2_count",
                "top_failure_3",
                "top_failure_3_count",
                "query_ids",
                "divergent_query_ids",
            ],
        )
        writer.writeheader()
        for row in behavior_rows:
            for name in profile_names:
                item = row["profile_summary"][name]
                top_failure_modes = list(item["top_failure_modes"])
                writer.writerow(
                    {
                        "behavior": row["behavior"],
                        "profile": name,
                        "profile_label": item["label"],
                        "query_count": row["query_count"],
                        "divergent_query_count": row["divergent_query_count"],
                        "best_profile_query_wins": item["best_profile_query_wins"],
                        "pass_at_1_count": item["pass_at_1_count"],
                        "pass_at_1_rate": item["pass_at_1_rate"],
                        "scene_objective_at_1_count": item["scene_objective_at_1_count"],
                        "scene_objective_at_1_rate": item["scene_objective_at_1_rate"],
                        "actor_objective_at_1_count": item["actor_objective_at_1_count"],
                        "actor_objective_at_1_rate": item["actor_objective_at_1_rate"],
                        "reference_objective_at_1_count": item["reference_objective_at_1_count"],
                        "reference_objective_at_1_rate": item["reference_objective_at_1_rate"],
                        "mean_event_iou": item["mean_event_iou"],
                        "mean_peak_error": item["mean_peak_error"],
                        "failure_query_count": item["failure_query_count"],
                        "top_failure_1": top_failure_modes[0]["name"] if len(top_failure_modes) > 0 else "",
                        "top_failure_1_count": top_failure_modes[0]["count"] if len(top_failure_modes) > 0 else "",
                        "top_failure_2": top_failure_modes[1]["name"] if len(top_failure_modes) > 1 else "",
                        "top_failure_2_count": top_failure_modes[1]["count"] if len(top_failure_modes) > 1 else "",
                        "top_failure_3": top_failure_modes[2]["name"] if len(top_failure_modes) > 2 else "",
                        "top_failure_3_count": top_failure_modes[2]["count"] if len(top_failure_modes) > 2 else "",
                        "query_ids": "|".join(row["query_ids"]),
                        "divergent_query_ids": "|".join(row["divergent_query_ids"]),
                    }
                )

    leaderboard_html_path.write_text(
        LEADERBOARD_TEMPLATE.render(comparison=comparison),
        encoding="utf-8",
    )
    behavior_html_path.write_text(
        BEHAVIOR_ANALYSIS_TEMPLATE.render(
            analysis=behavior_rows,
            profile_order=profile_names,
            profile_labels=profile_labels,
        ),
        encoding="utf-8",
    )
