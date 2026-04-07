from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from nusc_scene_agent.models import ValidatedCase


def _case_key(case: ValidatedCase) -> str:
    return "{0}:{1}".format(case.candidate.sample_token, case.candidate.instance_token)


def _merge_unique_strings(items: Sequence[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _selected_hypothesis(case: ValidatedCase) -> str:
    for marker in case.query.specific_keywords:
        if marker.startswith("planner:hybrid_selected:"):
            return marker.split("planner:hybrid_selected:", 1)[1]
    for marker, name in [
        ("planner:hybrid_merge", "hybrid_merge"),
        ("planner:llm_only", "llm"),
        ("planner:rule", "rule"),
    ]:
        if marker in case.query.specific_keywords:
            return name
    return "query"


def _entry_from_case(
    query_id: str,
    query_text: str,
    query_tags: Sequence[str],
    rank: int,
    case: ValidatedCase,
) -> Dict[str, object]:
    return {
        "case_key": _case_key(case),
        "scene_name": case.candidate.scene_name,
        "scene_token": case.candidate.scene_token,
        "sample_idx": case.candidate.sample_idx,
        "sample_token": case.candidate.sample_token,
        "instance_token": case.candidate.instance_token,
        "category_name": case.candidate.category_name,
        "category_group": case.candidate.category_group,
        "location": case.candidate.location,
        "passed": bool(case.passed),
        "validation_score": float(case.validation_score),
        "retrieval_score": float(case.candidate.retrieval_score),
        "min_distance_m": case.evidence.get("min_distance_m"),
        "min_ttc_s": case.evidence.get("min_ttc_s"),
        "source_query_ids": [query_id],
        "source_queries": [query_text],
        "source_query_tags": list(query_tags),
        "matched_behaviors": sorted([name for name, matched in case.behavior_matches.items() if matched]),
        "all_behaviors": sorted(case.query.behaviors),
        "selected_hypothesis": _selected_hypothesis(case),
        "planner_markers": sorted([item for item in case.query.specific_keywords if item.startswith("planner:")]),
        "rank_positions": [int(rank)],
        "map_available": bool(case.map_context.get("available")),
        "map_crosswalk": bool(case.map_context.get("actor_on_crosswalk_any")),
        "map_walkway": bool(case.map_context.get("actor_on_walkway_any")),
        "map_shared_lane": bool(case.map_context.get("shares_lane_at_anchor")),
        "map_actor_uses_ego_lane": bool(case.map_context.get("actor_uses_ego_lane_any")),
        "actor_track_start_sample_idx": case.actor_grounding.get("track_start_sample_idx"),
        "actor_track_end_sample_idx": case.actor_grounding.get("track_end_sample_idx"),
        "actor_track_frame_count": case.actor_grounding.get("track_frame_count"),
        "event_primary_behavior": case.event_localization.get("primary_behavior", ""),
        "event_start_sample_idx": case.event_localization.get("start_sample_idx"),
        "event_end_sample_idx": case.event_localization.get("end_sample_idx"),
        "event_peak_sample_idx": case.event_localization.get("peak_sample_idx"),
        "event_duration_s": case.event_localization.get("duration_s"),
        "figure_path": case.figure_path or "",
        "report_dir": case.report_dir or "",
        "notes": list(case.notes),
    }


def build_case_library(benchmark_results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}

    for result in benchmark_results:
        query_id = str(result.get("id") or result["query"].original_text)
        query_text = str(result["query"].original_text)
        query_spec = result.get("query_spec")
        query_tags = list(getattr(query_spec, "tags", []))
        for rank, case in enumerate(result["selected_cases"], start=1):
            entry = _entry_from_case(query_id, query_text, query_tags, rank, case)
            key = str(entry["case_key"])
            if key not in merged:
                merged[key] = entry
                continue

            current = merged[key]
            current["source_query_ids"] = _merge_unique_strings(
                list(current["source_query_ids"]) + list(entry["source_query_ids"])
            )
            current["source_queries"] = _merge_unique_strings(
                list(current["source_queries"]) + list(entry["source_queries"])
            )
            current["source_query_tags"] = _merge_unique_strings(
                list(current["source_query_tags"]) + list(entry["source_query_tags"])
            )
            current["matched_behaviors"] = _merge_unique_strings(
                list(current["matched_behaviors"]) + list(entry["matched_behaviors"])
            )
            current["all_behaviors"] = _merge_unique_strings(
                list(current["all_behaviors"]) + list(entry["all_behaviors"])
            )
            current["planner_markers"] = _merge_unique_strings(
                list(current["planner_markers"]) + list(entry["planner_markers"])
            )
            current["rank_positions"] = sorted(set(list(current["rank_positions"]) + list(entry["rank_positions"])))
            current["map_crosswalk"] = bool(current["map_crosswalk"] or entry["map_crosswalk"])
            current["map_walkway"] = bool(current["map_walkway"] or entry["map_walkway"])
            current["map_shared_lane"] = bool(current["map_shared_lane"] or entry["map_shared_lane"])
            current["map_actor_uses_ego_lane"] = bool(
                current["map_actor_uses_ego_lane"] or entry["map_actor_uses_ego_lane"]
            )
            current["notes"] = _merge_unique_strings(list(current["notes"]) + list(entry["notes"]))

            if float(entry["validation_score"]) > float(current["validation_score"]):
                for field in [
                    "scene_name",
                    "scene_token",
                    "sample_idx",
                    "sample_token",
                    "instance_token",
                    "category_name",
                    "category_group",
                    "location",
                    "passed",
                    "validation_score",
                    "retrieval_score",
                    "min_distance_m",
                    "min_ttc_s",
                    "map_available",
                    "actor_track_start_sample_idx",
                    "actor_track_end_sample_idx",
                    "actor_track_frame_count",
                    "event_primary_behavior",
                    "event_start_sample_idx",
                    "event_end_sample_idx",
                    "event_peak_sample_idx",
                    "event_duration_s",
                    "figure_path",
                    "report_dir",
                ]:
                    current[field] = entry[field]

    entries = list(merged.values())
    entries.sort(
        key=lambda item: (
            bool(item["passed"]),
            len(item["source_query_ids"]),
            float(item["validation_score"]),
        ),
        reverse=True,
    )
    return entries


def write_case_library(entries: Sequence[Dict[str, object]], output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "case_library.json"
    csv_path = output_dir / "case_library.csv"
    md_path = output_dir / "case_library_summary.md"

    json_path.write_text(json.dumps(list(entries), indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "case_key",
        "scene_name",
        "sample_idx",
        "category_name",
        "location",
        "passed",
        "validation_score",
        "retrieval_score",
        "min_distance_m",
        "min_ttc_s",
        "source_query_ids",
        "source_queries",
        "source_query_tags",
        "matched_behaviors",
        "all_behaviors",
        "selected_hypothesis",
        "planner_markers",
        "rank_positions",
        "map_available",
        "map_crosswalk",
        "map_walkway",
        "map_shared_lane",
        "map_actor_uses_ego_lane",
        "actor_track_start_sample_idx",
        "actor_track_end_sample_idx",
        "actor_track_frame_count",
        "event_primary_behavior",
        "event_start_sample_idx",
        "event_end_sample_idx",
        "event_peak_sample_idx",
        "event_duration_s",
        "figure_path",
        "report_dir",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            for key in [
                "source_query_ids",
                "source_queries",
                "matched_behaviors",
                "all_behaviors",
                "planner_markers",
                "rank_positions",
            ]:
                row[key] = "|".join(str(item) for item in row.get(key, []))
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    summary_lines = [
        "# Case Library Summary",
        "",
        "- Unique cases: {0}".format(len(entries)),
        "",
        "| Rank | Scene | Sample | Actor | Hypothesis | Passed | Score | Query Hits | Queries |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for rank, entry in enumerate(entries[:20], start=1):
        summary_lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6:.2f} | {7} | {8} |".format(
                rank,
                entry["scene_name"],
                entry["sample_idx"],
                entry["category_name"],
                entry.get("selected_hypothesis", "query"),
                entry["passed"],
                float(entry["validation_score"]),
                len(entry["source_query_ids"]),
                ", ".join(entry["source_query_ids"]),
            )
        )

    md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
