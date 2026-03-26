from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence


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


def default_benchmark_profiles(include_llm: bool = True) -> List[Dict[str, str]]:
    if include_llm:
        return [dict(item) for item in DEFAULT_BENCHMARK_PROFILES]
    return [dict(DEFAULT_BENCHMARK_PROFILES[0])]


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _top_taxonomy_label(taxonomy: Dict[str, object]) -> str:
    label_distribution = list(taxonomy.get("label_distribution") or [])
    if not label_distribution:
        return "none"
    return str(label_distribution[0].get("name") or "none")


def build_benchmark_comparison(profile_runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    profiles: List[Dict[str, object]] = []
    query_table: Dict[str, Dict[str, object]] = {}

    for run in profile_runs:
        output_dir = Path(run["output_dir"]).resolve()
        metrics = _load_json(output_dir / "benchmark_metrics.json")
        taxonomy = _load_json(output_dir / "hard_case_taxonomy.json")

        overview = dict(metrics.get("overview") or {})
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

    return {
        "overview": {
            "profile_count": len(profiles),
            "query_count": len(query_comparison),
            "profile_names": profile_name_order,
            "signal_divergence_count": sum(1 for row in query_comparison if row["signal_divergence"]),
        },
        "profiles": profiles,
        "deltas_vs_baseline": deltas_vs_baseline,
        "query_comparison": query_comparison,
    }


def write_benchmark_comparison(comparison: Dict[str, object], output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_profile_comparison.json"
    md_path = output_dir / "benchmark_profile_comparison_summary.md"

    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    profile_rows = list(comparison.get("profiles") or [])
    query_rows = list(comparison.get("query_comparison") or [])

    lines = [
        "# Benchmark Profile Comparison",
        "",
        "- Profiles: {0}".format(comparison["overview"]["profile_count"]),
        "- Queries: {0}".format(comparison["overview"]["query_count"]),
        "- Queries with planner signal divergence: {0}".format(comparison["overview"]["signal_divergence_count"]),
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
                "## Planner Signal Divergence",
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
