from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence

from jinja2 import Template


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
    Pass@1: {{ metrics.overview.pass_at_1_count }}/{{ metrics.overview.query_count }} ({{ "%.1f"|format(metrics.overview.pass_at_1_rate * 100.0) }}%) |
    Pass@K: {{ metrics.overview.pass_at_k_count }}/{{ metrics.overview.query_count }} ({{ "%.1f"|format(metrics.overview.pass_at_k_rate * 100.0) }}%)
  </div>

  <h2>Overview</h2>
  <table>
    <tbody>
      <tr><th>Selected Cases</th><td>{{ metrics.overview.selected_case_count }}</td></tr>
      <tr><th>Passed Selected Cases</th><td>{{ metrics.overview.passed_selected_case_count }}</td></tr>
      <tr><th>Unique Cases</th><td>{{ metrics.overview.unique_case_count }}</td></tr>
      <tr><th>Unique Passed Cases</th><td>{{ metrics.overview.unique_passed_case_count }}</td></tr>
      <tr><th>Unique Locations</th><td>{{ metrics.overview.unique_locations }}</td></tr>
      <tr><th>Mean Best Score</th><td>{{ "%.2f"|format(metrics.overview.mean_best_validation_score) }}</td></tr>
      <tr><th>Median Best Score</th><td>{{ "%.2f"|format(metrics.overview.median_best_validation_score) }}</td></tr>
    </tbody>
  </table>

  <h2>Query Breakdown</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Pass@1</th><th>Pass@K</th><th>Passed</th><th>Selected</th><th>Best Score</th><th>Actors</th><th>Behaviors</th>
      </tr>
    </thead>
    <tbody>
    {% for row in metrics.query_metrics %}
      <tr>
        <td>{{ row.id }}</td>
        <td>{{ row.pass_at_1 }}</td>
        <td>{{ row.pass_at_k }}</td>
        <td>{{ row.passed_count }}</td>
        <td>{{ row.selected_count }}</td>
        <td>{{ "%.2f"|format(row.best_validation_score) }}</td>
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
        <th>Behavior</th><th>Queries</th><th>Pass@1</th><th>Pass@K</th><th>Avg Best Score</th>
      </tr>
    </thead>
    <tbody>
    {% for row in metrics.behavior_coverage %}
      <tr>
        <td>{{ row.name }}</td>
        <td>{{ row.query_count }}</td>
        <td>{{ row.pass_at_1_count }}</td>
        <td>{{ row.pass_at_k_count }}</td>
        <td>{{ "%.2f"|format(row.avg_best_validation_score) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Actor Coverage</h2>
  <table>
    <thead>
      <tr>
        <th>Actor</th><th>Queries</th><th>Pass@1</th><th>Pass@K</th><th>Avg Best Score</th>
      </tr>
    </thead>
    <tbody>
    {% for row in metrics.actor_coverage %}
      <tr>
        <td>{{ row.name }}</td>
        <td>{{ row.query_count }}</td>
        <td>{{ row.pass_at_1_count }}</td>
        <td>{{ row.pass_at_k_count }}</td>
        <td>{{ "%.2f"|format(row.avg_best_validation_score) }}</td>
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
            group["best_scores"].append(float(row["best_validation_score"]))

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
                "avg_best_validation_score": round(mean(best_scores), 2) if best_scores else 0.0,
            }
        )

    coverage_rows.sort(
        key=lambda item: (
            int(item["pass_at_k_count"]),
            float(item["avg_best_validation_score"]),
            -len(str(item["name"])),
        ),
        reverse=True,
    )
    return coverage_rows


def build_benchmark_metrics(
    benchmark_results: Sequence[Dict[str, object]],
    case_library_entries: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    query_metrics: List[Dict[str, object]] = []

    for result in benchmark_results:
        query_spec = result.get("query_spec")
        selected_cases = list(result.get("selected_cases") or [])
        passed_count = sum(1 for case in selected_cases if case.passed)
        best_case = max(selected_cases, key=lambda item: float(item.validation_score), default=None)

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
                "selected_count": len(selected_cases),
                "passed_count": passed_count,
                "pass_at_1": bool(selected_cases and selected_cases[0].passed),
                "pass_at_k": bool(passed_count > 0),
                "best_validation_score": float(best_case.validation_score) if best_case is not None else 0.0,
                "best_scene_name": str(best_case.candidate.scene_name) if best_case is not None else "",
                "best_sample_idx": int(best_case.candidate.sample_idx) if best_case is not None else -1,
                "locations": sorted({str(case.candidate.location) for case in selected_cases}),
            }
        )

    query_count = len(query_metrics)
    pass_at_1_count = sum(1 for row in query_metrics if row["pass_at_1"])
    pass_at_k_count = sum(1 for row in query_metrics if row["pass_at_k"])
    best_scores = [float(row["best_validation_score"]) for row in query_metrics]
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
            "unique_locations": len(location_distribution),
            "mean_best_validation_score": round(mean(best_scores), 2) if best_scores else 0.0,
            "median_best_validation_score": round(median(best_scores), 2) if best_scores else 0.0,
        },
        "query_metrics": query_metrics,
        "behavior_coverage": _group_coverage(query_metrics, field_name="behaviors", empty_label="none"),
        "actor_coverage": _group_coverage(query_metrics, field_name="actors", empty_label="any"),
        "tag_coverage": _group_coverage(query_metrics, field_name="tags", empty_label="untagged"),
        "location_distribution": location_distribution,
    }
    return metrics


def write_benchmark_metrics(metrics: Dict[str, object], output_dir: Path) -> None:
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
        "- Pass@1: {0}/{1} ({2:.1%})".format(
            overview["pass_at_1_count"],
            overview["query_count"],
            overview["pass_at_1_rate"],
        ),
        "- Pass@K: {0}/{1} ({2:.1%})".format(
            overview["pass_at_k_count"],
            overview["query_count"],
            overview["pass_at_k_rate"],
        ),
        "- Selected cases: {0}".format(overview["selected_case_count"]),
        "- Passed selected cases: {0}".format(overview["passed_selected_case_count"]),
        "- Unique cases: {0}".format(overview["unique_case_count"]),
        "- Unique passed cases: {0}".format(overview["unique_passed_case_count"]),
        "- Unique locations: {0}".format(overview["unique_locations"]),
        "- Mean best validation score: {0:.2f}".format(overview["mean_best_validation_score"]),
        "",
        "## Query Breakdown",
        "",
        "| Query ID | Pass@1 | Pass@K | Passed | Selected | Best Score | Actors | Behaviors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in metrics["query_metrics"]:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5:.2f} | {6} | {7} |".format(
                row["id"],
                row["pass_at_1"],
                row["pass_at_k"],
                row["passed_count"],
                row["selected_count"],
                row["best_validation_score"],
                ", ".join(row["actors"]) or "any",
                ", ".join(row["behaviors"]) or "none",
            )
        )

    lines.extend(
        [
            "",
            "## Behavior Coverage",
            "",
            "| Behavior | Queries | Pass@1 | Pass@K | Avg Best Score |",
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
                row["avg_best_validation_score"],
            )
        )

    lines.extend(
        [
            "",
            "## Actor Coverage",
            "",
            "| Actor | Queries | Pass@1 | Pass@K | Avg Best Score |",
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
                row["avg_best_validation_score"],
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
