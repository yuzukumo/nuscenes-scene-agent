from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from jinja2 import Template
import numpy as np

from nusc_scene_agent.models import ParsedQuery, ValidatedCase
from nusc_scene_agent.visualization import render_case_figure


SUMMARY_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>nuScenes Scene Mining Summary</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 32px; background: #f5f1ea; color: #1f1f1f; }
    h1, h2 { margin-bottom: 0.4rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; background: white; }
    th, td { border-bottom: 1px solid #ddd4c7; padding: 10px 12px; text-align: left; }
    th { background: #ede5d8; }
    a { color: #c04a00; text-decoration: none; }
    .meta { color: #665d54; margin-bottom: 18px; }
  </style>
</head>
<body>
  <h1>nuScenes Scene Mining Summary</h1>
  <div class="meta">Query: {{ query.original_text }}</div>
  {% if agent_trace %}
  <h2>Agent Trace</h2>
  <table>
    <tbody>
      <tr><th>Mode</th><td>{{ agent_trace.mode }}</td></tr>
      <tr><th>Selected Hypothesis</th><td>{{ agent_trace.selected_hypothesis }}</td></tr>
      <tr><th>Retrieval Score Profile</th><td>{{ agent_trace.retrieval_score_profile }}</td></tr>
      <tr><th>Selection Policy</th><td>{{ agent_trace.selection_policy }}</td></tr>
    </tbody>
  </table>
  <table>
    <thead>
      <tr>
        <th>Hypothesis</th>
        <th>Selected</th>
        <th>Actors</th>
        <th>Positions</th>
        <th>Behaviors</th>
        <th>Passed</th>
        <th>Best Validation Quality</th>
      </tr>
    </thead>
    <tbody>
    {% for trace_row in agent_trace.hypotheses %}
      <tr>
        <td>{{ trace_row.name }}</td>
        <td>{{ trace_row.selected }}</td>
        <td>{{ ", ".join(trace_row.query.category_groups) or "none" }}</td>
        <td>{{ ", ".join(trace_row.query.positions) or "none" }}</td>
        <td>{{ ", ".join(trace_row.query.behaviors) or "none" }}</td>
        <td>{{ trace_row.passed_count }}</td>
        <td>{{ "%.2f"|format(trace_row.best_validation_quality_score) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Scene</th>
        <th>Sample</th>
        <th>Actor</th>
        <th>Passed</th>
        <th>Validation Quality</th>
        <th>Min Distance</th>
        <th>Case</th>
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row.rank }}</td>
        <td>{{ row.scene_name }}</td>
        <td>{{ row.sample_idx }}</td>
        <td>{{ row.actor }}</td>
        <td>{{ row.passed }}</td>
        <td>{{ "%.2f"|format(row.validation_quality_score) }}</td>
        <td>{{ "%.2f"|format(row.min_distance_m) }}</td>
        <td><a href="{{ row.case_dir }}/case.md">case.md</a></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "query"


def _normalize_agent_trace(agent_trace: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Add canonical quality fields while preserving trace compatibility."""
    if not agent_trace:
        return agent_trace
    normalized = dict(agent_trace)
    hypotheses = []
    for raw_item in list(agent_trace.get("hypotheses") or []):
        item = dict(raw_item)
        quality = item.get("best_validation_quality_score")
        if quality is None:
            quality = item.get("best_validation_score", 0.0)
        item["best_validation_quality_score"] = float(quality or 0.0)
        item.setdefault("best_validation_score", item["best_validation_quality_score"])
        hypotheses.append(item)
    normalized["hypotheses"] = hypotheses
    return normalized


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _render_agent_trace_markdown(agent_trace: Optional[Dict[str, object]]) -> List[str]:
    if not agent_trace:
        return []

    lines = [
        "## Agent Trace",
        "",
        "- Mode: {0}".format(agent_trace.get("mode", "unknown")),
        "- Selected hypothesis: {0}".format(agent_trace.get("selected_hypothesis", "unknown")),
        "- Retrieval score profile: {0}".format(agent_trace.get("retrieval_score_profile", "unknown")),
        "- Selection policy: {0}".format(agent_trace.get("selection_policy", "")),
        "",
        "| Hypothesis | Selected | Actors | Positions | Behaviors | Passed | Best Validation Quality |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in agent_trace.get("hypotheses") or []:
        query_payload = dict(item.get("query") or {})
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6:.2f} |".format(
                item.get("name", "query"),
                item.get("selected", False),
                ", ".join(query_payload.get("category_groups") or []) or "none",
                ", ".join(query_payload.get("positions") or []) or "none",
                ", ".join(query_payload.get("behaviors") or []) or "none",
                item.get("passed_count", 0),
                float(
                    item.get("best_validation_quality_score")
                    if item.get("best_validation_quality_score") is not None
                    else item.get("best_validation_score")
                    or 0.0
                ),
            )
        )
    lines.append("")
    return lines


def _render_case_markdown(
    case: ValidatedCase,
    figure_name: str,
    agent_trace: Optional[Dict[str, object]] = None,
) -> str:
    lines = [
        "# Case Report",
        "",
        "- Query: {0}".format(case.query.original_text),
        "- Scene: {0}".format(case.candidate.scene_name),
        "- Sample Index: {0}".format(case.candidate.sample_idx),
        "- Actor: {0}".format(case.candidate.category_name),
        "- Validation Quality Score: {0:.2f}".format(case.validation_quality_score),
        "- Validation Gate: {0}".format(case.gate_decision.get("status", "pass" if case.passed else "fail")),
        "- Gate Score: {0:.3f}".format(case.gate_score),
        "",
        "## Evidence",
        "",
    ]
    lines.extend(_render_agent_trace_markdown(agent_trace))

    for key, value in sorted(case.evidence.items()):
        lines.append("- {0}: {1}".format(key, value))

    if case.behavior_matches:
        lines.extend(["", "## Behavior Checks", ""])
        for name, matched in sorted(case.behavior_matches.items()):
            lines.append("- {0}: {1}".format(name, matched))

    if case.map_context:
        lines.extend(["", "## Map Context", ""])
        for key in [
            "ego_in_lane_anchor",
            "ego_on_drivable_anchor",
            "actor_on_crosswalk_any",
            "actor_on_walkway_any",
            "actor_uses_ego_lane_any",
            "shares_lane_at_anchor",
            "ego_closest_lane",
            "actor_closest_lane_anchor",
        ]:
            if key in case.map_context:
                lines.append("- {0}: {1}".format(key, case.map_context[key]))

    if case.actor_grounding:
        lines.extend(["", "## Actor Grounding", ""])
        for key in [
            "role",
            "category_name",
            "category_group",
            "instance_token",
            "anchor_sample_idx",
            "track_start_sample_idx",
            "track_end_sample_idx",
            "track_frame_count",
        ]:
            if key in case.actor_grounding:
                lines.append("- {0}: {1}".format(key, case.actor_grounding[key]))

    if case.event_localization:
        lines.extend(["", "## Event Localization", ""])
        for key in [
            "primary_behavior",
            "start_sample_idx",
            "end_sample_idx",
            "peak_sample_idx",
            "start_t_sec",
            "end_t_sec",
            "peak_t_sec",
            "duration_s",
            "frame_count",
            "anchor_within_window",
        ]:
            if key in case.event_localization:
                lines.append("- {0}: {1}".format(key, case.event_localization[key]))

    lines.extend(["", "## Notes", ""])
    lines.extend(["- {0}".format(item) for item in case.notes])
    lines.extend(["", "## Figure", "", "![evidence]({0})".format(figure_name), ""])
    return "\n".join(lines)


def write_query_report(
    query: ParsedQuery,
    cases: Sequence[ValidatedCase],
    output_dir: Path,
    agent_trace: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    agent_trace = _normalize_agent_trace(agent_trace)

    if agent_trace:
        (output_dir / "query_trace.json").write_text(
            json.dumps(_json_safe(agent_trace), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for rank, case in enumerate(cases, start=1):
        case_dir = output_dir / "rank_{0:02d}_{1}_sample_{2:03d}".format(
            rank,
            slugify(case.candidate.scene_name),
            case.candidate.sample_idx,
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        figure_path = render_case_figure(case, case_dir / "evidence.png")
        case.figure_path = str(figure_path)
        case.report_dir = str(case_dir)

        case_payload = _json_safe(case.summary_dict())
        if agent_trace:
            case_payload["agent_trace"] = _json_safe(agent_trace)
        (case_dir / "case.json").write_text(json.dumps(case_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (case_dir / "case.md").write_text(
            _render_case_markdown(case, figure_path.name, agent_trace=agent_trace),
            encoding="utf-8",
        )

        rows.append(
            {
                "rank": rank,
                "scene_name": case.candidate.scene_name,
                "sample_idx": case.candidate.sample_idx,
                "actor": case.candidate.category_name,
                "passed": case.passed,
                "validation_quality_score": case.validation_quality_score,
                # Keep the legacy row key for consumers of existing reports.
                "validation_score": case.validation_quality_score,
                "min_distance_m": case.evidence.get("min_distance_m", 0.0),
                "case_dir": case_dir.name,
            }
        )

    summary_md = [
        "# Query Summary",
        "",
        "- Query: {0}".format(query.original_text),
        "- Returned cases: {0}".format(len(rows)),
        "",
    ]
    summary_md.extend(_render_agent_trace_markdown(agent_trace))
    summary_md.extend(
        [
            "| Rank | Scene | Sample | Actor | Passed | Quality | Min Distance |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        summary_md.append(
            "| {rank} | {scene_name} | {sample_idx} | {actor} | {passed} | {validation_quality_score:.2f} | {min_distance_m:.2f} |".format(
                **row
            )
        )

    (output_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")
    (output_dir / "summary.html").write_text(
        SUMMARY_TEMPLATE.render(query=query.to_dict(), rows=rows, agent_trace=_json_safe(agent_trace)),
        encoding="utf-8",
    )
    return rows
