from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml

from nusc_scene_agent.counterfactual_benchmark import (
    _case_semantics,
    _canonical_positive_query,
    _diverse_anchor_entries,
    _paraphrase_positive_query,
    _thresholds_for_entry,
)


def _scenario_queries_for_entry(entry: Dict[str, object]) -> List[Dict[str, object]]:
    semantics = _case_semantics(entry)
    case_key = str(entry["case_key"])
    group = "scenario_" + case_key.replace(":", "_")
    thresholds = _thresholds_for_entry(entry)
    reference_scene_names = [str(entry["scene_name"])] if entry.get("scene_name") else []
    reference_instance_tokens = [str(entry["instance_token"])] if entry.get("instance_token") else []
    reference_event_sample_range = []
    if entry.get("event_start_sample_idx") is not None and entry.get("event_end_sample_idx") is not None:
        reference_event_sample_range = [
            int(entry["event_start_sample_idx"]),
            int(entry["event_end_sample_idx"]),
        ]
    reference_peak_sample_idx = (
        int(entry["event_peak_sample_idx"]) if entry.get("event_peak_sample_idx") is not None else None
    )
    actor = str(semantics["actor"])
    behaviors = list(semantics["behaviors"]) or (
        [str(semantics["primary_behavior"])] if semantics["primary_behavior"] != "proximity" else []
    )
    positions = list(semantics["positions"])
    risk_terms = list(semantics["risk_terms"])

    variants = [
        {
            "variant_type": "scenario_canonical",
            "query_text": _canonical_positive_query(semantics),
            "tags": ["scenario_mining", "canonical"] + list(behaviors or ["proximity"]),
        },
        {
            "variant_type": "scenario_paraphrase",
            "query_text": _paraphrase_positive_query(semantics),
            "tags": ["scenario_mining", "paraphrase"] + list(behaviors or ["proximity"]),
            "risk_terms": list(dict.fromkeys(risk_terms + ["risky"])),
        },
    ]

    specs: List[Dict[str, object]] = []
    for variant in variants:
        specs.append(
            {
                "id": "{0}_{1}".format(group, variant["variant_type"]),
                "description": "{0} {1}".format(entry["scene_name"], variant["variant_type"]),
                "top_k": 3,
                "candidate_pool": 12,
                "apply_query_overrides": False,
                "tags": list(variant["tags"]),
                "reference_case_keys": [case_key],
                "reference_scene_names": list(reference_scene_names),
                "reference_instance_tokens": list(reference_instance_tokens),
                "reference_event_sample_range": list(reference_event_sample_range),
                "reference_peak_sample_idx": reference_peak_sample_idx,
                "expect_match": True,
                "benchmark_group": group,
                "variant_type": str(variant["variant_type"]),
                "query": {
                    "natural_language": str(variant["query_text"]),
                    "actors": [actor],
                    "positions": list(positions),
                    "behaviors": list(behaviors),
                    "risk_terms": list(variant.get("risk_terms", risk_terms)),
                    "thresholds": dict(thresholds),
                },
            }
        )
    return specs


def generate_scenario_mining_benchmark_from_case_library(
    case_library_path: Path,
    output_path: Path,
    max_cases: int = 8,
) -> Dict[str, object]:
    source_case_library = str(case_library_path)
    case_library_path = case_library_path.resolve()
    output_path = output_path.resolve()
    entries = json.loads(case_library_path.read_text(encoding="utf-8"))
    anchors = _diverse_anchor_entries(entries, max_cases=max_cases)

    queries: List[Dict[str, object]] = []
    for entry in anchors:
        queries.extend(_scenario_queries_for_entry(entry))

    payload = {
        "metadata": {
            "generator": "scenario_mining_benchmark_generator_v1",
            "source_case_library": source_case_library,
            "anchor_case_count": len(anchors),
            "query_count": len(queries),
        },
        "queries": queries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload["metadata"]
