from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def _sorted_unique(items: Sequence[str]) -> List[str]:
    return sorted({str(item) for item in items if item})


def _safe_float(value: object, default: float = float("inf")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _classify_hard_case_taxonomy(entry: Dict[str, object]) -> Dict[str, str]:
    difficulty_label = str(entry.get("difficulty_label") or "")
    behaviors = set(_sorted_unique(entry.get("all_behaviors", [])))
    matched_behaviors = set(_sorted_unique(entry.get("matched_behaviors", [])))
    missing_behaviors = sorted(behaviors - matched_behaviors)
    min_distance_m = _safe_float(entry.get("min_distance_m"))
    min_ttc_s = _safe_float(entry.get("min_ttc_s"))
    has_crosswalk_support = bool(entry.get("map_crosswalk") or entry.get("map_walkway"))
    has_lane_support = bool(entry.get("map_shared_lane") or entry.get("map_actor_uses_ego_lane"))
    query_hit_count = int(entry.get("query_hit_count") or 0)

    if difficulty_label == "shared" or query_hit_count > 1:
        if query_hit_count >= 3:
            return {"taxonomy_group": "overlap", "taxonomy_label": "high_overlap_anchor"}
        return {"taxonomy_group": "overlap", "taxonomy_label": "multi_query_overlap"}

    if not bool(entry.get("passed")):
        if missing_behaviors:
            primary_behavior = missing_behaviors[0]
            if primary_behavior == "crossing" and not has_crosswalk_support:
                return {"taxonomy_group": "behavior_gap", "taxonomy_label": "crossing_without_map_support"}
            if primary_behavior in {"cut_in", "oncoming", "stopped_lead"} and not has_lane_support:
                return {"taxonomy_group": "behavior_gap", "taxonomy_label": "lane_relation_mismatch"}
            return {"taxonomy_group": "behavior_gap", "taxonomy_label": "behavior_mismatch"}
        if min_distance_m >= 12.0:
            return {"taxonomy_group": "geometry_gap", "taxonomy_label": "weak_proximity"}
        if min_ttc_s >= 5.0:
            return {"taxonomy_group": "dynamics_gap", "taxonomy_label": "weak_ttc_signal"}
        if "crossing" in behaviors and not has_crosswalk_support:
            return {"taxonomy_group": "map_gap", "taxonomy_label": "crossing_map_gap"}
        if behaviors.intersection({"cut_in", "oncoming", "stopped_lead"}) and not has_lane_support:
            return {"taxonomy_group": "map_gap", "taxonomy_label": "lane_support_gap"}
        return {"taxonomy_group": "other", "taxonomy_label": "low_confidence_failure"}

    if difficulty_label == "borderline":
        if min_distance_m >= 8.0:
            return {"taxonomy_group": "borderline", "taxonomy_label": "borderline_proximity"}
        if min_ttc_s >= 3.5:
            return {"taxonomy_group": "borderline", "taxonomy_label": "borderline_ttc"}
        if "crossing" in behaviors and not has_crosswalk_support:
            return {"taxonomy_group": "borderline", "taxonomy_label": "borderline_crossing_map_support"}
        if behaviors.intersection({"cut_in", "oncoming", "stopped_lead"}) and not has_lane_support:
            return {"taxonomy_group": "borderline", "taxonomy_label": "borderline_lane_support"}
        return {"taxonomy_group": "borderline", "taxonomy_label": "borderline_score"}

    return {"taxonomy_group": "other", "taxonomy_label": "other_hard_case"}


def build_query_splits(benchmark_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_behavior: Dict[str, List[str]] = defaultdict(list)
    by_actor: Dict[str, List[str]] = defaultdict(list)
    by_position: Dict[str, List[str]] = defaultdict(list)
    by_tag: Dict[str, List[str]] = defaultdict(list)
    pass_at_1: List[str] = []
    pass_at_k: List[str] = []
    gap_queries: List[str] = []

    for result in benchmark_results:
        query_id = str(result.get("id") or result["query"].original_text)
        query_spec = result.get("query_spec")
        selected_cases = list(result.get("selected_cases") or [])
        if selected_cases and selected_cases[0].passed:
            pass_at_1.append(query_id)
        if any(case.passed for case in selected_cases):
            pass_at_k.append(query_id)
        else:
            gap_queries.append(query_id)

        actors = list(getattr(query_spec, "actors", []))
        behaviors = list(getattr(query_spec, "behaviors", []))
        positions = list(getattr(query_spec, "positions", []))
        tags = list(getattr(query_spec, "tags", []))

        for name in actors or ["any"]:
            by_actor[str(name)].append(query_id)
        for name in behaviors or ["none"]:
            by_behavior[str(name)].append(query_id)
        for name in positions or ["any"]:
            by_position[str(name)].append(query_id)
        for name in tags or ["untagged"]:
            by_tag[str(name)].append(query_id)

    return {
        "overview": {
            "query_count": len(benchmark_results),
            "pass_at_1_query_ids": sorted(pass_at_1),
            "pass_at_k_query_ids": sorted(pass_at_k),
            "gap_query_ids": sorted(gap_queries),
        },
        "by_behavior": {key: sorted(value) for key, value in sorted(by_behavior.items())},
        "by_actor": {key: sorted(value) for key, value in sorted(by_actor.items())},
        "by_position": {key: sorted(value) for key, value in sorted(by_position.items())},
        "by_tag": {key: sorted(value) for key, value in sorted(by_tag.items())},
    }


def build_hard_cases(case_library_entries: Sequence[Dict[str, object]], borderline_score: float = 85.0) -> List[Dict[str, object]]:
    hard_cases: List[Dict[str, object]] = []

    for entry in case_library_entries:
        validation_score = float(entry["validation_score"])
        query_hit_count = len(entry.get("source_query_ids", []))
        if not entry.get("passed"):
            difficulty_label = "failed"
        elif validation_score < borderline_score:
            difficulty_label = "borderline"
        elif query_hit_count > 1:
            difficulty_label = "shared"
        else:
            continue

        difficulty_score = round(
            (20.0 if difficulty_label == "failed" else 0.0)
            + max(0.0, borderline_score - validation_score)
            + 2.0 * float(query_hit_count),
            2,
        )
        hard_case = {
            "case_key": entry["case_key"],
            "scene_name": entry["scene_name"],
            "sample_idx": int(entry["sample_idx"]),
            "category_name": entry["category_name"],
            "location": entry["location"],
            "passed": bool(entry["passed"]),
            "validation_score": validation_score,
            "difficulty_label": difficulty_label,
            "difficulty_score": difficulty_score,
            "query_hit_count": query_hit_count,
            "source_query_ids": list(entry.get("source_query_ids", [])),
            "matched_behaviors": list(entry.get("matched_behaviors", [])),
            "all_behaviors": list(entry.get("all_behaviors", [])),
            "min_distance_m": entry.get("min_distance_m"),
            "min_ttc_s": entry.get("min_ttc_s"),
            "map_shared_lane": bool(entry.get("map_shared_lane")),
            "map_crosswalk": bool(entry.get("map_crosswalk")),
            "map_walkway": bool(entry.get("map_walkway")),
            "map_actor_uses_ego_lane": bool(entry.get("map_actor_uses_ego_lane")),
            "figure_path": entry.get("figure_path", ""),
            "report_dir": entry.get("report_dir", ""),
            "notes": list(entry.get("notes", [])),
        }
        hard_case.update(_classify_hard_case_taxonomy(hard_case))
        hard_cases.append(hard_case)

    hard_cases.sort(
        key=lambda item: (
            float(item["difficulty_score"]),
            int(item["query_hit_count"]),
            -float(item["validation_score"]),
        ),
        reverse=True,
    )
    return hard_cases


def build_hard_case_taxonomy(hard_cases: Sequence[Dict[str, object]]) -> Dict[str, object]:
    group_stats: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"count": 0, "failed_count": 0, "borderline_count": 0, "shared_count": 0}
    )
    label_stats: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "taxonomy_group": "",
            "count": 0,
            "failed_count": 0,
            "borderline_count": 0,
            "shared_count": 0,
            "example_queries": [],
        }
    )

    overview = {"hard_case_count": len(hard_cases), "failed_count": 0, "borderline_count": 0, "shared_count": 0}

    for entry in hard_cases:
        difficulty_label = str(entry.get("difficulty_label") or "")
        taxonomy_group = str(entry.get("taxonomy_group") or "other")
        taxonomy_label = str(entry.get("taxonomy_label") or "other_hard_case")

        group = group_stats[taxonomy_group]
        group["count"] += 1

        label = label_stats[taxonomy_label]
        label["taxonomy_group"] = taxonomy_group
        label["count"] += 1
        label["example_queries"] = _sorted_unique(
            list(label["example_queries"]) + list(entry.get("source_query_ids", []))
        )[:6]

        if difficulty_label == "failed":
            overview["failed_count"] += 1
            group["failed_count"] += 1
            label["failed_count"] += 1
        elif difficulty_label == "borderline":
            overview["borderline_count"] += 1
            group["borderline_count"] += 1
            label["borderline_count"] += 1
        elif difficulty_label == "shared":
            overview["shared_count"] += 1
            group["shared_count"] += 1
            label["shared_count"] += 1

    group_distribution = [
        {
            "name": name,
            "count": int(stats["count"]),
            "failed_count": int(stats["failed_count"]),
            "borderline_count": int(stats["borderline_count"]),
            "shared_count": int(stats["shared_count"]),
        }
        for name, stats in group_stats.items()
    ]
    group_distribution.sort(
        key=lambda item: (int(item["count"]), int(item["failed_count"]), int(item["borderline_count"])),
        reverse=True,
    )

    label_distribution = [
        {
            "name": name,
            "taxonomy_group": str(stats["taxonomy_group"]),
            "count": int(stats["count"]),
            "failed_count": int(stats["failed_count"]),
            "borderline_count": int(stats["borderline_count"]),
            "shared_count": int(stats["shared_count"]),
            "example_queries": list(stats["example_queries"]),
        }
        for name, stats in label_stats.items()
    ]
    label_distribution.sort(
        key=lambda item: (int(item["count"]), int(item["failed_count"]), int(item["borderline_count"])),
        reverse=True,
    )

    return {
        "overview": overview,
        "group_distribution": group_distribution,
        "label_distribution": label_distribution,
    }


def write_benchmark_exports(
    benchmark_results: Sequence[Dict[str, object]],
    case_library_entries: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    query_splits = build_query_splits(benchmark_results)
    hard_cases = build_hard_cases(case_library_entries)
    hard_case_taxonomy = build_hard_case_taxonomy(hard_cases)

    (output_dir / "benchmark_splits.json").write_text(
        json.dumps(query_splits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "hard_cases.json").write_text(
        json.dumps(hard_cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "hard_case_taxonomy.json").write_text(
        json.dumps(hard_case_taxonomy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    split_lines = [
        "# Benchmark Splits",
        "",
        "- Query count: {0}".format(query_splits["overview"]["query_count"]),
        "- Pass@1 query ids: {0}".format(", ".join(query_splits["overview"]["pass_at_1_query_ids"]) or "none"),
        "- Gap query ids: {0}".format(", ".join(query_splits["overview"]["gap_query_ids"]) or "none"),
        "",
        "## Behavior Splits",
        "",
        "| Behavior | Query IDs |",
        "| --- | --- |",
    ]
    for key, values in query_splits["by_behavior"].items():
        split_lines.append("| {0} | {1} |".format(key, ", ".join(values)))

    split_lines.extend(
        [
            "",
            "## Actor Splits",
            "",
            "| Actor | Query IDs |",
            "| --- | --- |",
        ]
    )
    for key, values in query_splits["by_actor"].items():
        split_lines.append("| {0} | {1} |".format(key, ", ".join(values)))

    split_lines.extend(
        [
            "",
            "## Position Splits",
            "",
            "| Position | Query IDs |",
            "| --- | --- |",
        ]
    )
    for key, values in query_splits["by_position"].items():
        split_lines.append("| {0} | {1} |".format(key, ", ".join(values)))

    (output_dir / "benchmark_splits_summary.md").write_text("\n".join(split_lines) + "\n", encoding="utf-8")

    hard_lines = [
        "# Hard Cases",
        "",
        "- Count: {0}".format(len(hard_cases)),
        "",
        "| Rank | Label | Taxonomy | Scene | Sample | Actor | Passed | Score | Query Hits | Queries |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, entry in enumerate(hard_cases[:30], start=1):
        hard_lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7:.2f} | {8} | {9} |".format(
                idx,
                entry["difficulty_label"],
                entry["taxonomy_label"],
                entry["scene_name"],
                entry["sample_idx"],
                entry["category_name"],
                entry["passed"],
                entry["validation_score"],
                entry["query_hit_count"],
                ", ".join(entry["source_query_ids"]),
            )
        )
    (output_dir / "hard_cases_summary.md").write_text("\n".join(hard_lines) + "\n", encoding="utf-8")

    taxonomy_lines = [
        "# Hard Case Taxonomy",
        "",
        "- Hard cases: {0}".format(hard_case_taxonomy["overview"]["hard_case_count"]),
        "- Failed: {0}".format(hard_case_taxonomy["overview"]["failed_count"]),
        "- Borderline: {0}".format(hard_case_taxonomy["overview"]["borderline_count"]),
        "- Shared: {0}".format(hard_case_taxonomy["overview"]["shared_count"]),
        "",
        "## Taxonomy Groups",
        "",
        "| Group | Count | Failed | Borderline | Shared |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in hard_case_taxonomy["group_distribution"]:
        taxonomy_lines.append(
            "| {0} | {1} | {2} | {3} | {4} |".format(
                row["name"],
                row["count"],
                row["failed_count"],
                row["borderline_count"],
                row["shared_count"],
            )
        )

    taxonomy_lines.extend(
        [
            "",
            "## Taxonomy Labels",
            "",
            "| Label | Group | Count | Failed | Borderline | Shared | Example Queries |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in hard_case_taxonomy["label_distribution"]:
        taxonomy_lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(
                row["name"],
                row["taxonomy_group"],
                row["count"],
                row["failed_count"],
                row["borderline_count"],
                row["shared_count"],
                ", ".join(row["example_queries"]) or "none",
            )
        )
    (output_dir / "hard_case_taxonomy_summary.md").write_text("\n".join(taxonomy_lines) + "\n", encoding="utf-8")

    csv_fields = [
        "case_key",
        "scene_name",
        "sample_idx",
        "category_name",
        "location",
        "passed",
        "validation_score",
        "difficulty_label",
        "taxonomy_group",
        "taxonomy_label",
        "difficulty_score",
        "query_hit_count",
        "source_query_ids",
        "matched_behaviors",
        "all_behaviors",
        "min_distance_m",
        "min_ttc_s",
        "map_shared_lane",
        "map_crosswalk",
        "map_walkway",
        "map_actor_uses_ego_lane",
        "figure_path",
        "report_dir",
    ]
    with (output_dir / "hard_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for entry in hard_cases:
            row = dict(entry)
            row["source_query_ids"] = "|".join(_sorted_unique(row.get("source_query_ids", [])))
            row["matched_behaviors"] = "|".join(_sorted_unique(row.get("matched_behaviors", [])))
            row["all_behaviors"] = "|".join(_sorted_unique(row.get("all_behaviors", [])))
            writer.writerow({field: row.get(field, "") for field in csv_fields})
